# !pip install openai
from openai import OpenAI

client = OpenAI(
#        base_url='http://192.168.0.22:11434/v1',
        base_url='http://localhost:8080/v1',
        api_key="ollama"
)

prompt = '很高興認識你，你叫什麼名字'

response = client.chat.completions.create(
    model='gemma4',        
    messages=[
        {
            'role': 'user',
            'content': prompt
        },
    ]
)
result = response.choices[0].message.content
print(result)
