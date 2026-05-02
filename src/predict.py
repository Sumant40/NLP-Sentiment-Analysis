from pathlib import Path
import joblib
from src.preprocessing import clean_text

BASE_DIR = Path(__file__).resolve().parents[1]

model = joblib.load('models/lr_model.pkl')
vectorizer = joblib.load('models/tfidf.pkl')

def predict(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    
    return {
        "label": "Positive" if pred == 1 else "Negative",
        "confidence": float(max(prob))
    }