from collections import Counter
import re

from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from nltk.stem import PorterStemmer

app = Flask(__name__)

# Preserve negation words because "not good" differs from "good".
STOP_WORDS = set(ENGLISH_STOP_WORDS) - {"not", "no", "nor"}
STEMMER = PorterStemmer()
MIN_TRAINING_SAMPLES = 8

def preprocess_text(text):
    """Lowercase, remove punctuation, tokenize, remove stopwords, and stem."""
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = re.findall(r"\b[a-z0-9]+\b", text)
    tokens = [STEMMER.stem(token) for token in tokens if token not in STOP_WORDS]
    return tokens

# Small illustrative seed data; replace/expand with labelled real data for production.
SENTIMENT_TRAINING = [
    ("delicious fresh food and quick delivery", "Positive"),
    ("excellent meal and friendly service", "Positive"),
    ("hot tasty pizza arrived quickly", "Positive"),
    ("great burger and perfect packaging", "Positive"),
    ("very happy with my order", "Positive"),
    ("wonderful food will order again", "Positive"),
    ("fresh ingredients and generous portions", "Positive"),
    ("great value and friendly driver", "Positive"),
    ("food quality was very good", "Positive"),
    ("the food was good and fresh", "Positive"),
    ("very good meal and excellent service", "Positive"),
    ("delicious food arrived on time", "Positive"),
    ("friendly delivery partner and tasty food", "Positive"),
    ("my food arrived late and rider was not friendly", "Negative"),
    ("delivery was late and driver was rude", "Negative"),
    ("my order arrived late and the food was cold", "Negative"),
    ("cold food and terrible taste", "Negative"),
    ("delivery was late and order missing", "Negative"),
    ("awful meal and poor service", "Negative"),
    ("food spilled and arrived cold", "Negative"),
    ("app charged me twice", "Negative"),
    ("raw chicken and stale fries", "Negative"),
    ("wrong order and disappointing experience", "Negative"),
    ("never ordering from here again", "Negative"),
]

CATEGORY_TRAINING = [
    ("driver is late and delivery not here", "Delivery"),
    ("order never arrived from driver", "Delivery"),
    ("delivery tracking shows no movement", "Delivery"),
    ("courier went to the wrong address", "Delivery"),
    ("delivery was delayed again today", "Delivery"),
    ("driver cancelled my order", "Delivery"),
    ("pizza is cold and burger is raw", "Food Quality"),
    ("food tastes stale and smells bad", "Food Quality"),
    ("drink spilled and fries are soggy", "Food Quality"),
    ("chicken was undercooked and cold", "Food Quality"),
    ("meal was burnt and salty", "Food Quality"),
    ("food quality was terrible", "Food Quality"),
    ("app crashes during checkout", "App"),
    ("payment failed in the app", "App"),
    ("cannot login to my account", "App"),
    ("promo code is not accepted", "App"),
    ("application keeps freezing", "App"),
    ("app charged me twice", "App"),
    ("thank you restaurant for a lovely meal", "General"),
    ("I enjoyed the service and food", "General"),
    ("I have a question about the menu", "General"),
    ("the experience was nice overall", "General"),
    ("please pass my thanks to the staff", "General"),
    ("I want to share general feedback", "General"),
]

def make_pipeline(training_data):
    texts = [" ".join(preprocess_text(text)) for text, _ in training_data]
    labels = [label for _, label in training_data]
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    pipeline.fit(texts, labels)
    return pipeline

sentiment_model = make_pipeline(SENTIMENT_TRAINING)
category_model = make_pipeline(CATEGORY_TRAINING)

# In-memory store: reviews are retained until the Flask server restarts.
reviews = []
next_id = 1

def adjust_sentiment_for_clear_phrases(raw_text, model_sentiment):
    """
    Transparent phrase override for clear examples in this educational demo.
    A production system should use a larger labelled dataset and held-out evaluation.
    """
    text = re.sub(r"[^a-z0-9\s']", " ", str(raw_text or "").lower())
    text = re.sub(r"\s+", " ", text).strip()

    # Check negative phrases first so "not good" is not treated as positive.
    negative_phrases = (
        "not good", "not great", "not tasty", "not fresh", "not friendly",
        "never arrived", "never delivered", "charged me twice", "very late",
        "arrived late", "delivery was late", "food was cold", "arrived cold",
        "raw chicken", "terrible", "awful", "disappoint", "poor service",
        "poor quality", "rude", "missing", "burnt", "stale", "soggy",
        "crashed", "payment failed", "wrong order", "spilled", "unacceptable",
        "bad taste", "bad food", "cold and late", "late and cold"
    )
    if any(phrase in text for phrase in negative_phrases):
        return "Negative"

    positive_phrases = (
        "very good", "really good", "so good", "food was good",
        "quality was good", "quality was very good", "delicious",
        "excellent", "amazing", "fantastic", "loved the", "love the",
        "fresh and tasty", "fresh and delicious", "arrived on time",
        "quick delivery", "friendly service", "friendly delivery",
        "great service", "great food", "perfect meal", "happy with"
    )
    if any(phrase in text for phrase in positive_phrases):
        return "Positive"

    return model_sentiment


def classify(text):
    cleaned_tokens = preprocess_text(text)
    cleaned_text = " ".join(cleaned_tokens)
    if len(SENTIMENT_TRAINING) < MIN_TRAINING_SAMPLES or len(CATEGORY_TRAINING) < MIN_TRAINING_SAMPLES:
        raise ValueError(
            f"Not enough training samples. At least {MIN_TRAINING_SAMPLES} labelled samples "
            "are required for each classifier."
        )
    if not cleaned_text:
        raise ValueError("The review has no content left after preprocessing. Please enter more detail.")
    sentiment = sentiment_model.predict([cleaned_text])[0]
    sentiment = adjust_sentiment_for_clear_phrases(text, sentiment)
    category = category_model.predict([cleaned_text])[0]
    return cleaned_tokens, sentiment, category

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "sentiment_training_samples": len(SENTIMENT_TRAINING),
                    "category_training_samples": len(CATEGORY_TRAINING)})

@app.get("/api/reviews")
def get_reviews():
    return jsonify({"reviews": list(reversed(reviews)), "total": len(reviews)})

@app.post("/api/classify")
def classify_review():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    if not text:
        return jsonify({"error": "Please enter a review to classify."}), 400
    try:
        tokens, sentiment, category = classify(text)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    return jsonify({
        "original_text": text,
        "processed_text": " ".join(tokens),
        "sentiment": sentiment,
        "category": category
    })

@app.post("/api/reviews")
def add_review():
    global next_id
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    if not text:
        return jsonify({"error": "Please enter a review before saving."}), 400
    try:
        tokens, sentiment, category = classify(text)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422

    item = {
        "id": next_id,
        "text": text,
        "processed_text": " ".join(tokens),
        "sentiment": sentiment,
        "category": category
    }
    next_id += 1
    reviews.append(item)
    return jsonify({"message": "Review classified and saved.", "review": item, "summary": build_summary()}), 201

def build_summary():
    sentiment_counts = {"Positive": 0, "Negative": 0}
    category_counts = {"Delivery": 0, "Food Quality": 0, "App": 0, "General": 0}
    words = []
    for review in reviews:
        sentiment_counts[review["sentiment"]] = sentiment_counts.get(review["sentiment"], 0) + 1
        category_counts[review["category"]] = category_counts.get(review["category"], 0) + 1
        words.extend(review["processed_text"].split())
    return {
        "total_reviews": len(reviews),
        "sentiment_counts": sentiment_counts,
        "category_counts": category_counts,
        "top_words": [{"word": word, "count": count} for word, count in Counter(words).most_common(5)]
    }

@app.get("/api/summary")
def summary():
    return jsonify(build_summary())

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
