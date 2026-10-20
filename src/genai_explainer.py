"""Grounded GenAI explanation using Gemini REST API. No secret is stored in source."""
import os, json, time
from dotenv import load_dotenv
import requests

load_dotenv()

def explain_with_gemini(facts: dict) -> tuple[str, float]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Copy .env.example to .env and add your key from Google AI Studio.")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    prompt = (
        "You are a cautious fraud-risk analyst assisting a human reviewer. Explain this transaction "
        "using only the supplied facts. Clearly separate observed facts from possible interpretations. "
        "Do not claim fraud is confirmed. Do not invent location, device, account history, or user intent. "
        "Give: (1) risk decision in plain language, (2) evidence, (3) uncertainty/limitations, "
        "(4) recommended next review step. Keep under 130 words. Facts JSON:\\n"
        + json.dumps(facts, ensure_ascii=False)
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}],
               "generationConfig": {"temperature": 0.2, "maxOutputTokens": 220}}
    start = time.perf_counter()
    response = requests.post(url, json=payload, timeout=25)
    elapsed_ms = (time.perf_counter() - start) * 1000
    if not response.ok:
        raise RuntimeError(f"Gemini API returned HTTP {response.status_code}: {response.text[:500]}")
    data = response.json()
    try:
        answer = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("Gemini returned no text. Check model availability and API quota.")
    return answer, elapsed_ms
