#!/usr/bin/env bash
# Run one measured profile at a time.
case "${1:-balanced}" in
  speed) exec /kaggle/working/artemis-bench/llama-server-sm75 -m /tmp/artemis-tools/model.gguf -ngl all --fit off -sm layer -ts 1,1 -mg 0 -fa on -ctk f16 -ctv f16 -c 4096 -b 512 -ub 128 -t 4 -np 1 --host 127.0.0.1 --port 18080 --no-webui --no-warmup --model-draft /tmp/artemis-tools/draft.gguf --spec-type draft-simple --spec-draft-ngl all --spec-draft-n-max 3 --ctx-checkpoints 0 --cache-ram 0 -lv 4 ;;
  context) exec /kaggle/working/artemis-bench/llama-server-sm75 -m /tmp/artemis-tools/model.gguf -ngl all --fit off -sm tensor -ts 1,1 -mg 0 -fa on -ctk q4_0 -ctv q4_0 -c 262144 -b 2048 -ub 128 -t 4 -np 1 --host 127.0.0.1 --port 18080 --no-webui --no-warmup --override-tensor token_embd.weight=CPU --ctx-checkpoints 0 --cache-ram 0 --no-context-shift ;;
  balanced) exec /kaggle/working/artemis-bench/llama-server-sm75 -m /tmp/artemis-tools/model.gguf -ngl all --fit off -sm tensor -ts 1,1 -mg 0 -fa on -ctk q8_0 -ctv q8_0 -c 98304 -b 2048 -ub 128 -t 4 -np 1 --host 127.0.0.1 --port 18080 --no-webui --no-warmup --ctx-checkpoints 0 --cache-ram 0 --no-context-shift -lv 4 ;;
  *) echo "Choose speed, context, or balanced" >&2; exit 2 ;;
esac
