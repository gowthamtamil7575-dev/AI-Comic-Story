import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Initialize Google GenAI client
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

app = FastAPI(title="EduGenie API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StudyRequest(BaseModel):
    topic: str
    content: str
    mode: str  # "overview", "deep_dive", or "quiz"
    academic_level: str

@app.post("/api/generate-study-material")
async def generate_study_material(req: StudyRequest):
    try:
        prompt = f"""
        Act as an expert AI tutor. Generate comprehensive learning materials for the following request:

        - Subject/Topic: {req.topic}
        - Academic Level: {req.academic_level}
        - Focus Mode: {req.mode}
        - Source Content/Notes: {req.content if req.content else 'N/A (Generate based on general topic knowledge)'}

        Structure your output into clear markdown sections:
        1. ### 📌 Key Concept Summary
        2. ### 🧠 Core Learning Points
        3. ### 📇 Flashcards (Term -> Definition)
        4. ### ✏️ Practice Quiz Questions (With Answers & Explanations)
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="You are EduGenie, an encouraging, clear, and pedagogically effective AI learning assistant.",
                temperature=0.3,
            ),
        )

        return {"success": True, "materials": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)