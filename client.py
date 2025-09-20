import requests

with open("Hostel_Affidavit_Men_2024-Chennai_Updated.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/summarize", files=files)

print(response.json())
# Replace with your question
question_text = "exit"

# Send POST request to /ask
response = requests.post(
    "http://localhost:8000/ask",
    data={"question": question_text}  # use 'data' for form fields
)

print(response.json())
