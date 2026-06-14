curl http://localhost:8080/v1/chat/completions -d '{ 
  "messages": [{ 
    "role": "user",
    "content": "Say hello to me"
  }]
}'
