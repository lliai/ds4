  直接运行流程如下。

  先从 C4 生成校准文本，默认使用 Hugging Face allenai/c4 的 en/train streaming，需要当前 Python 环境有 datasets：

  python3 gguf-tools/expert-prune/build_c4_prune_dataset.py \
    --out /tmp/ds4-c4-prune.txt \
    --max-docs 512 \
    --max-chars-per-doc 8192

  如果你已经有本地 C4 JSONL，例如每行有 text 字段：

  python3 gguf-tools/expert-prune/build_c4_prune_dataset.py \
    --source jsonl \
    --input /path/to/c4.jsonl \
    --out /tmp/ds4-c4-prune.txt \
    --max-docs 512

  然后收集 OMP 统计并生成每层 7/8 keep mask。Flash 是 224/256，Pro 是 336/384：

  CUDA_VISIBLE_DEVICES=4 ./ds4 --cuda \
    -m ds4flash.gguf \
    --expert-prune-dataset /tmp/ds4-c4-prune.txt \
    --expert-prune-out /tmp/ds4-c4-keep7of8.json \
    --expert-prune-keep-ratio 0.875 \
    --expert-prune-max-tokens 131072 \

  CUDA_VISIBLE_DEVICES=4 ./ds4 --cuda \
    -m ds4flash.gguf \
    --expert-mask /tmp/ds4-c4-keep7of8.json \
    --nothink \
    -n 128 \
    -p "Explain expert pruning in one paragraph."

  评测、测速和服务也可以带同一个 mask：

  ./ds4-eval --cuda -m ds4flash.gguf --expert-mask /tmp/ds4-c4-keep7of8.json --questions 5 --plain
  ./ds4-bench --cuda -m ds4flash.gguf --expert-mask /tmp/ds4-c4-keep7of8.json --prompt-file speed-bench/promessi_sposi.txt
  ./ds4-server --cuda -m ds4flash.gguf --expert-mask /tmp/ds4-c4-keep7of8.json --ctx 32768

  验证结果：

  - python3 gguf-tools/expert-prune/build_c4_prune_dataset.py --help 通过。
  - 用 README.md 做本地 text-source smoke，成功生成 DS4_IMATRIX_PROMPT 分隔的校准文件。
  - make cpu 通过；保留了仓库已有的非阻塞 warning。
  - CUDA_HOME=/usr/local/cuda make cuda CUDA_ARCH=sm_90 通过。第一次直接 make cuda 失败是因为环境里的 CUDA_HOME 展开成了错误的 /usr/local/cuda:/
    usr/local/cuda/...，显式指定后正常。
  - 实际 CUDA smoke 已跑通：用 1 个 prompt、16 tokens 生成 /tmp/ds4-prune-mask-smoke.json，校验结果是 ds4-expert-mask-v1 43 256 224 43 [224]，即
    43 层每层保留 224 个专家。
  - 用这个 smoke mask 做 4 token 推理也通过，输出开头为 Hello! How can，说明 --expert-mask 加载和 masked CUDA 路由路径能启动。

  当前仍然不是“结构化缩小 GGUF”。如果下一步目标是减少磁盘体积或显已经把这条 DS4 专家剪枝链路整理成可直接跑的形态，并做了实际 CUDA smoke 验证。

  关键点先说清楚：当前实现是 shape-preserving runtime mask pruning，不是物理删除 GGUF 里的专家张量。也就是说它会生成 ds4-expert-mask-v1 JSON，每
  层保留 7/8 专家，推理时让路由避开被剪掉的专家；它不会减小 GGUF 文件体积，也不会改 deepseek4.expert_count。这符合当前 DS4 loader/router 的硬约
  束，真正结构化删专家还需要另一套 GGUF metadata、tensor layout、hash routing、CPU/CUDA kernel 的协同修改。

  改动内容：

  - 在 ds4.c:14201 中已有的 OMP-style 收集/选择链路现在可以收集每层 routed MoE 输出统计，按 --expert-prune-keep-ratio 生成每层固定数量 keep-
    list；0.875 对 Flash 就是每层 224/256。
  - 在 ds4_cli.c:188 暴露了校准入口：--expert-prune-dataset、--expert-prune-out、--expert-prune-keep-ratio，默认 keep ratio 是 0.875。
  - 在 gguf-tools/expert-prune/build_c4_prune_dataset.py:1 新增 C4 校准文件生成脚本，支持 Hugging Face streaming C4，也支持本地 JSONL/text。
  - 在 gguf-tools/expert-prune/README.md:1 写了完整流程命令；gguf-tools/README.md:124 和 README.md:101 也加了入口说明。
  - 把 --expert-mask FILE 也接到了 ds4-eval、ds4-bench、ds4-server、ds4-agent，例如 ds4_eval.c:1497、ds4_server.c:11369。

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

  验证结果：

  - python3 gguf-tools/expert-prune/build_c4_prune_dataset.py --help 通过。
  - 用 README.md 做本地 text-source smoke，成功生成 marker-separated 校准文件。
  - make cpu 通过；仍有仓库已有的非阻塞 warning。
  - 初次 CUDA build 因环境里的 CUDA_HOME 展开成 /usr/local/cuda:/usr/local/cuda/... 失败；显式指定后 CUDA_HOME=/usr/local/cuda make cuda
    CUDA_ARCH=sm_90 通过。
  - 实际 CUDA OMP mask smoke 已跑通：/tmp/ds4-prune-mask-smoke.json 显示 43 层、256 专家、每层 224 keep，所有层 keep 数都是 [224]。
  - 实际带 mask 的 CUDA 4-token 推理 smoke 已跑通，模型加载了 mask 并输出了 Hello! How can。

  当前工作区仍有未提交改动，新增目录是 gguf-tools/expert-prune/。