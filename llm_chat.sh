curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4","messages":[{"role":"user","content":"Hi!"}],"max_tokens":32}' \
  | python3 -m json.tool
