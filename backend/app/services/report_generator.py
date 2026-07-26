import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def generate_medical_report(prediction, confidence):
    prompt = f"""
You are an AI medical assistant.

Brain MRI AI Result

Prediction: {prediction}
Confidence: {confidence:.2f}%

Return ONLY valid JSON.

Example:

{{
    "diagnosis_summary": "...",
    "clinical_explanation": "...",
    "confidence_interpretation": "...",
    "recommended_next_steps": "...",
    "medical_disclaimer": "..."
}}

Do not use markdown.
Do not use ```json.
Return only the JSON object.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    try:
        return json.loads(response.text)
    except Exception:
        return {
            "diagnosis_summary": "Unable to generate report.",
            "clinical_explanation": response.text,
            "confidence_interpretation": "",
            "recommended_next_steps": "",
            "medical_disclaimer": "AI-generated report."
        }