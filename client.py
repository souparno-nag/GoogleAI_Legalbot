import requests

with open("res.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/summarize", files=files)

print(response.json())
