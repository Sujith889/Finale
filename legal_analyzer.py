import os
import re
from difflib import SequenceMatcher
from dateparser.search import search_dates
import docx
import fitz  # PyMuPDF
from transformers import pipeline

# Initialize models once here (or you can do lazy loading in frontend)
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
sentiment_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

clause_keywords = {
    "Obligation": ["shall", "must", "agree to"],
    "Rights": ["reserves the right", "may"],
    "Confidentiality": ["confidential", "non-disclosure"],
    "Risk": ["liable", "indemnify", "at your own risk"]
}

boilerplate_templates = [
    "This Agreement shall be governed by the laws of the State of",
    "The parties agree to the following terms and conditions"
]

def split_into_clauses(text):
    return re.split(r'(?<=\.)\s+(?=[A-Z])|\n|\d+\.\s+', text.strip())

def classify_clause(clause):
    for category, keywords in clause_keywords.items():
        for kw in keywords:
            if kw.lower() in clause.lower():
                return category
    return "Other"

def grade_clause(clause):
    risk_keywords = ["liable", "indemnify", "terminate", "breach"]
    score = sum(1 for kw in risk_keywords if kw in clause.lower())
    importance = "High" if score >= 2 else "Medium" if score == 1 else "Low"
    return importance, score

def extract_timeline(clause):
    dates = search_dates(clause)
    return [(d[0], d[1].strftime("%Y-%m-%d")) for d in dates] if dates else []

def detect_boilerplate(clause):
    for template in boilerplate_templates:
        if SequenceMatcher(None, clause.lower(), template.lower()).ratio() > 0.8:
            return True
    return False

def read_txt(file):
    return file.read().decode("utf-8")

def read_docx(file):
    doc = docx.Document(file)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def read_pdf(file):
    text = ""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

def analyze_tone(text):
    text = text[:512]  # truncate for speed
    sentiment_result = sentiment_classifier(text)[0]['label'].lower()
    emotion_scores = emotion_classifier(text)[0]
    top_emotion = max(emotion_scores, key=lambda x: x['score'])['label'].lower()
    return sentiment_result, top_emotion

def summarize(text):
    max_len = 1024
    summary = summarizer(text[:max_len])[0]['summary_text']
    return summary

def rewrite_clause(clause):
    clause_lower = clause.lower()
    if "indemnify" in clause_lower:
        return "Consider softening indemnity to mutual responsibility."
    if "liable" in clause_lower:
        return "Clarify or limit liability language."
    return "Clause appears clear."

# Main analysis function for one document
def analyze_document(text):
    clauses = split_into_clauses(text)
    results = []
    for i, clause in enumerate(clauses, 1):
        if not clause.strip():
            continue
        category = classify_clause(clause)
        importance, risk_score = grade_clause(clause)
        dates = extract_timeline(clause)
        is_boilerplate = detect_boilerplate(clause)
        sentiment, tone = analyze_tone(clause)
        rewrite = rewrite_clause(clause)

        results.append({
            "clause_num": i,
            "clause_text": clause,
            "category": category,
            "importance": importance,
            "risk_score": risk_score,
            "dates": dates,
            "boilerplate": is_boilerplate,
            "sentiment": sentiment,
            "tone": tone,
            "rewrite_suggestion": rewrite
        })
    return results
