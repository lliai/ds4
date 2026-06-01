你可以直接这样跑 C4 版本：

python3 gguf-tools/expert-prune/build_c4_prune_dataset.py \
  --out /tmp/ds4-c4-prune.txt \
  --max-docs 512 \
  --max-chars-per-doc 8192

如果你本地已经有 C4 JSONL，每行有 text 字段：

python3 gguf-tools/expert-prune/build_c4_prune_dataset.py \
  --source jsonl \
  --input /path/to/c4.jsonl \
  --out /tmp/ds4-c4-prune.txt \
  --max-docs 512

生成每层保留 7/8 专家的 mask：

CUDA_VISIBLE_DEVICES=4 ./ds4 --cuda \
  -m ds4flash.gguf \
  --expert-prune-dataset /tmp/ds4-c4-prune.txt \
  --expert-prune-out /tmp/ds4-c4-keep7of8.json \
  --expert-prune-keep-ratio 0.875 \
  --expert-prune-max-tokens 131072 \
  --ctx 32768

用 mask 推理：

CUDA_VISIBLE_DEVICES=4 ./ds4 --cuda \
  -m ds4flash.gguf \
  --expert-mask /tmp/ds4-c4-keep7of8.json \
  --nothink \
  -n 128 \
  -p "Explain OMP expert pruning briefly."

也可以评测或服务：

CUDA_VISIBLE_DEVICES=4 ./ds4-eval --cuda -m ds4flash.gguf --expert-mask /tmp/ds4-c4-keep7of8.json --plain
CUDA_VISIBLE_DEVICES=4 ./ds4-server --cuda -m ds4flash.gguf --expert-mask /tmp/ds4-c4-keep7of8.json --ctx 32768
CUDA_VISIBLE_DEVICES=4 ./ds4-bench --cuda -m ds4flash.gguf --expert-mask /tmp/ds4-c4-keep7of8.json --prompt-file speed-bench/
promessi_sposi.txt --ctx-max 2048


部署 ds4-server 提供 HTTP 推理端口：

```sh
setsid env \
  CUDA_VISIBLE_DEVICES=4 \
  DS4_LOCK_FILE=/data1/ldz/ds4/gguf-tools/expert-prune/sweep-runs/20260526-012444/locks/ds4-gpu-4.lock \
  /data1/ldz/ds4/ds4-server \
    --cuda \
    -m /data1/ldz/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
    --expert-mask /data1/ldz/ds4/gguf-tools/expert-prune/sweep-runs/20260526-012444/keep-036/expert-mask.json \
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
