# Food Delivery Review Intelligence System

A full-stack educational NLP project for Section C of the NLP assessment.

## Features
- Responsive frontend dashboard.
- Flask backend REST API.
- Text preprocessing: lowercase, punctuation removal, tokenization, English stopword removal, and Porter stemming.
- TF-IDF + Logistic Regression sentiment classification (Positive/Negative).
- TF-IDF + Logistic Regression complaint classification (Delivery/Food Quality/App/General).
- Classify-only and classify-and-save actions.
- Summary report: total reviews, sentiment counts, issue-category counts, and top five content words.
- Graceful validation for empty text and minimum training sample requirements.
- In-memory storage for the current server session.

## Project structure
```text
Food_Delivery_Review_Intelligence_System/
├── app.py
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
└── static/
    ├── app.js
    └── style.css
```

## Run on Windows
1. Install Python 3.10+.
2. Open this folder in VS Code.
3. In the terminal:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   python app.py
   ```
4. Open `http://127.0.0.1:5000` in your browser.

## Run on macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```
Then open `http://127.0.0.1:5000`.

## API endpoints
- `GET /api/health` — API and training sample counts.
- `GET /api/reviews` — list saved reviews.
- `POST /api/classify` — classify without saving; JSON body: `{"text":"The delivery was late"}`.
- `POST /api/reviews` — classify and save a review.
- `GET /api/summary` — report totals and top content words.

## Notes for assessment
- The starter training examples are small illustrative examples for demonstrating the pipeline. Model predictions may not be reliable on real customer data. Expand and validate training data before real deployment.
- Reviews are stored in memory and will reset when the Flask server restarts. A database such as SQLite can be added for persistent storage.
- The app includes an explicit minimum-training-sample check. The included sample data already meets the threshold; if the training sets are reduced below the minimum, classification returns a clear error instead of crashing.
- For the assignment, capture screenshots of the dashboard, a classification result, review history, and summary report.


## Sentiment prediction note
The demo now includes extra negative training examples and a transparent phrase-based correction for obvious complaint language (for example, "arrived late" and "not friendly"). This makes the supplied example behave more sensibly, but it is not a substitute for training and validating on a larger labelled dataset. A production version should test false positives/negatives on a held-out dataset.


## Sentiment demo tuning
The sentiment demo includes extra positive training examples and handles explicit positive phrases (such as "food quality was very good") as well as clear negative phrases (such as "not good" and "arrived late"). This heuristic is intended for an educational demo; real deployments should use a larger labelled dataset and evaluate errors on held-out examples.
