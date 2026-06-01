## Linux运行

你可以直接这样跑 C4 版本：

```sh
python3 gguf-tools/expert-prune/build_c4_prune_dataset.py \
  --out gguf-tools/expert-prune/ds4-c4-prune.txt \
  --max-docs 512 \
  --max-chars-per-doc 8192
```

如果你本地已经有 C4 JSONL，每行有 text 字段：

```sh
python3 gguf-tools/expert-prune/build_c4_prune_dataset.py \
  --source jsonl \
  --input /path/to/c4.jsonl \
  --out gguf-tools/expert-prune/ds4-c4-prune.txt \
  --max-docs 512
```

生成每层保留 36% 专家的 mask：

```sh
CUDA_VISIBLE_DEVICES=4 ./ds4 --cuda \
  -m ds4flash.gguf \
  --expert-prune-dataset gguf-tools/expert-prune/ds4-c4-prune.txt \
  --expert-prune-out gguf-tools/expert-prune/ds4-c4-keep036.json \
  --expert-prune-keep-ratio 0.36 \
  --expert-prune-max-tokens 131072 \
  --ctx 32768
```

用 mask 推理：

```sh
CUDA_VISIBLE_DEVICES=4 ./ds4 --cuda \
  -m ds4flash.gguf \
  --expert-mask gguf-tools/expert-prune/ds4-c4-keep036.json \
  --nothink \
  -n 128 \
  -p "Explain OMP expert pruning briefly."
```


部署 ds4-server 提供 HTTP 推理端口：

```sh
setsid env \
  CUDA_VISIBLE_DEVICES=4 \
  /data1/ldz/ds4/ds4-server \
    --cuda \
    -m ./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
    --expert-mask gguf-tools/expert-prune/ds4-c4-keep036.json \
    --ctx 4096 \
    -n 1024 \
    --host 127.0.0.1 \
    --port 8000 \
    --kv-disk-dir /tmp/ds4-kv-gpu4 \
    --kv-disk-space-mb 4096 \
    > /tmp/ds4-server-gpu4-8000.log 2>&1 < /dev/null &
```

参数说明：

- `--host 127.0.0.1 --port 8000` 只开放本机访问；如果要让其他机器访问，改成 `--host 0.0.0.0`，浏览器跨域调用再加 `--cors`。
- `--ctx 4096` 对齐前面单次推理 smoke 的上下文大小；需要更长上下文时可以增大，但会增加 KV 显存/内存占用。
- `-n 1024` 是客户端不传 `max_tokens` 时的默认输出上限；请求里仍可用 `max_tokens` 覆盖。
- `--kv-disk-dir /tmp/ds4-kv-gpu4` 开启磁盘 KV cache，长提示词重复请求时可以复用前缀。
- server 本身没有 `--nothink` 启动参数；OpenAI chat 请求里用 `model: "deepseek-chat"` 或传 `think: false` 来选择非 thinking 模式。

验证 server 是否启动：

```sh
curl -s http://127.0.0.1:8000/v1/models
```

发送一次 OpenAI-compatible chat completion 请求：

```sh
curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-chat",
    "messages": [
      {
        "role": "user",
        "content": "Please answer in two short sentences: introduce yourself and say one thing you can help with."
      }
    ],
    "max_tokens": 80,
    "temperature": 0
  }'
```

停止服务时用启动后打印的 PID：

```sh
kill <pid>
```

## macOS Metal 运行

在 macOS 上先构建 Metal 版本：

```sh
make
```

使用已有 expert mask 进行 Metal 推理：

```sh
./ds4 --metal \
  -m ds4flash.gguf \
  --expert-mask gguf-tools/expert-prune/ds4-c4-keep036.json \
  --nothink \
  -n 128 \
  -p "Explain OMP expert pruning briefly."
```

部署 Metal HTTP server：

```sh
setsid env \
  /path/to/ds4/ds4-server \
    --chdir /path/to/ds4 \
    --metal \
    -m ./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
    --expert-mask gguf-tools/expert-prune/ds4-c4-keep036.json \
    --ctx 4096 \
    -n 1024 \
    --host 127.0.0.1 \
    --port 8000 \
    --kv-disk-dir /tmp/ds4-kv-metal \
    --kv-disk-space-mb 4096 \
    > /tmp/ds4-server-metal-8000.log 2>&1 < /dev/null &
```

说明：Metal 后端只在 macOS Metal 构建中可用；Linux 当前应继续使用 CUDA 或 CPU 后端。
