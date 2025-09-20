from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from summarization import final_summary


app = FastAPI()

# Enable CORS so frontend on another domain can call it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/summarize")
async def get_summary(file: UploadFile = File(...)):
    contents = await file.read()

    with open("uploaded.pdf", "wb") as f:
        f.write(contents)

    summary_result = await final_summary("uploaded.pdf")
    return summary_result

