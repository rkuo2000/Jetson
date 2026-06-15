~/llama.cpp/build/bin/llama-server \
  -m ~/models/gemma-4-E2B-it-Q4_K_M.gguf \
  --mmproj ~/models/mmproj-gemma4-e2b-f16.gguf \
  -c 2048 \
  --image-min-tokens 70 --image-max-tokens 70 \
  --ubatch-size 512 --batch-size 512 \
  --host 0.0.0.0 --port 8080 \
  -ngl 99 --flash-attn on \
  --no-mmproj-offload --jinja -np 1
