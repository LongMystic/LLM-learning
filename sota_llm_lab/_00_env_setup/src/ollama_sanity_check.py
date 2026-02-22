import requests
import json

resp = requests.post(
    'http://localhost:11434/api/generate',
    json = {
        'model': "llama3.2",
        'prompt': "Giới thiệu ngắn về bản thân bạn. Trả lời bằng Tiếng Việt."
    },
    stream=True
)

for line in resp.iter_lines():
    if line:
        data = json.loads(line)
        print(data.get("response", ""), end="", flush=True)