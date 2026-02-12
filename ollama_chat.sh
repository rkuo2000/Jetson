curl http://10.15.120.169:11434/api/chat -H "Content-Type: application/json" -d '{
  "model": "qwen3-coder-next",
  "messages": [
    {
      "role": "user",
      "content": "Hello!"
    }
  ],
  "stream": false
}'
