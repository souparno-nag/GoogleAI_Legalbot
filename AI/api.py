import asyncio
from fastapi import FastAPI, Request
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

# Dummy summarization result (replace with your generated JSON later)
# summary_result = {
#     "title": "Hostel Affidavit Summary",
#     "key_points": [
#         "This document outlines hostel rules for male students.",
#         "Students must follow curfew regulations.",
#         "Parents/guardians are required to sign the affidavit."
#     ],
#     "risk_level": "Medium"
# }

@app.get("/summarize")
async def get_summary():
    summary_result = await final_summary("../Hostel_Affidavit_Men_2024-Chennai_Updated.pdf")
    return summary_result

