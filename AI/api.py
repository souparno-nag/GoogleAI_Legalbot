from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from summarization import final_summary
from qna import init_chat
import shutil
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store chatbot
GRAPH = None
BOT = None
CURRENT_FILE = None


@app.post("/summarize")
async def summarize_and_store(file: UploadFile = File(...)):
    global BOT, GRAPH, CURRENT_FILE
    # Save the uploaded PDF temporarily
    file_path = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    CURRENT_FILE = file_path

    # Generate summary
    summary_result = await final_summary(file_path)

    # Initialize chatbot for Q&A
    BOT, GRAPH = await init_chat(file_path)

    return {"summary": summary_result, "message": "File summarized and bot ready!"}


@app.post("/ask")
async def ask_question(question: str = Form(...)):
    global BOT, CURRENT_FILE
    if BOT is None or CURRENT_FILE is None:
        return {"error": "No file summarized yet. Please upload a PDF first via /summarize."}

    # Get answer from bot
    answer = await BOT.ask(question)
    return {"question": question, "answer": answer}
