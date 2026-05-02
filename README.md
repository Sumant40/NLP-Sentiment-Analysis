# Amazon Review Sentiment Classifier NLP Project

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![NLP](https://img.shields.io/badge/NLP-TF--IDF%20%7C%20DistilBERT-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-blue)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## Problem Statement

E-commerce platforms receive millions of product reviews daily. Manually
reading and categorising them for quality control, product feedback, and
customer satisfaction monitoring is not scalable. This project builds an
automated sentiment classifier that processes reviews at scale, helping
product and operations teams prioritise customer feedback efficiently.

**Business question:** Can we automatically classify product reviews as
positive or negative with high enough accuracy to replace manual review
tagging at scale?

---

## Dataset

- **Source:** Amazon Fine Food Reviews (Kaggle)
- **Link:** https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
- **Full size:** 568,454 reviews (2002â€“2012)
- **Used:** 20,000 reviews (stratified sample for speed)
- **Target variable:** Sentiment derived from star rating
  - Positive: 4-5 stars
  - Negative: 1-2 stars
  - Neutral (3 stars): removed from analysis

| Class    | Count  | % of sample |
|----------|--------|-------------|
| Positive | 16,842 | 84.2%       |
| Negative | 3,158  | 15.8%       |

> Class imbalance noted â€” F1-score used as primary metric, not accuracy.

---

## Project Structure

```text
nlp-sentiment-analysis/
|
|-- app/
|   |-- api.py                 # FastAPI backend for prediction requests
|   `-- streamlit_app.py       # Streamlit interface for interactive testing
|
|-- data/
|   |-- raw/
|   |   `-- Reviews.csv        # Original Amazon Fine Food Reviews data
|   `-- processed/
|       `-- cleaned_reviews.csv
|
|-- models/
|   |-- lr_model.pkl           # Saved Logistic Regression model
|   `-- tfidf.pkl              # Saved TF-IDF vectorizer
|
|-- notebooks/
|   |-- 01_data_cleaning.ipynb
|   |-- 02_eda.ipynb
|   |-- 03_model_training.ipynb
|   `-- 04_transformer_comparison.ipynb
|
|-- outputs/
|   `-- figures/
|       |-- class_distribution.png
|       |-- pos_wc.png
|       `-- neg_wc.png
|
|-- src/
|   |-- preprocessing.py       # Text cleaning
|   |-- features.py            # TF-IDF feature extraction
|   |-- train.py               # Model training
|   |-- evaluate.py            # Model evaluation
|   `-- predict.py             # Inference helper used by API and Streamlit
|
|-- requirements.txt
|-- requirements-dev.txt
|-- setup.py
`-- README.md
```

---

## Methodology

### 1. Text preprocessing pipeline

```text
Raw review text
    -> Convert to string
    -> Lowercase
    -> Remove HTML tags
    -> Expand n't to not
    -> Remove non-alphabetic characters
    -> Return cleaned text
```

Example:

```text
ORIGINAL: "This is <b>AMAZING</b>!! Best dog food I've ever bought. 5/5 stars!!!"
CLEANED:  "this is amazing best dog food ive ever bought  stars"
```

The preprocessing logic is implemented in `src/preprocessing.py`.

### 2. Feature extraction - TF-IDF

TF-IDF converts cleaned review text into numerical features that machine
learning models can use.

Settings used:
- `max_features=5000` - top 5,000 vocabulary terms
- `ngram_range=(1,2)` - single words and two-word phrases
- `min_df=5` - ignore terms appearing in fewer than 5 reviews

The vectorizer is built in `src/features.py` and saved as `models/tfidf.pkl`.

### 3. Model training

The main model is Logistic Regression trained on TF-IDF features.

Training details:
- `class_weight='balanced'` to handle class imbalance
- `max_iter=1000`
- `GridSearchCV` over `C=[0.1, 1, 5]`
- 3-fold cross-validation
- F1-score optimization

The trained model is saved as `models/lr_model.pkl`.

### 4. Evaluation strategy

Because the dataset is imbalanced, accuracy alone is misleading. F1-score,
precision, recall, and ROC-AUC are used to evaluate model performance.

### 5. Inference pipeline

The inference helper in `src/predict.py`:
- loads the saved Logistic Regression model
- loads the saved TF-IDF vectorizer
- cleans new review text
- converts it to TF-IDF features
- returns the predicted sentiment label and confidence score

---

## New Features Added

### FastAPI backend

`app/api.py` exposes a `/predict` endpoint that accepts review text and
returns the model prediction as JSON.

Run the backend:

```bash
uvicorn app.api:app --reload
```

Example request body:

```json
{
  "text": "This product is fresh and delicious."
}
```

Example response:

```json
{
  "label": "Positive",
  "confidence": 0.97
}
```

### Streamlit web app

`app/streamlit_app.py` provides a simple browser interface where users can
enter a review and instantly view the predicted sentiment.

Run the app:

```bash
streamlit run app/streamlit_app.py
```

---

## Results

Results from `notebooks/03_model_training.ipynb` using an 80/20 stratified
train-test split:

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | ROC-AUC |
|-------|----------|--------------------|-----------------|-------------|---------|
| Logistic Regression + TF-IDF | 91.0% | 0.92 | 0.91 | 0.91 | 0.954 |

Class-level performance:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Negative | 0.66 | 0.83 | 0.74 | 620 |
| Positive | 0.97 | 0.92 | 0.94 | 3,380 |

The transformer comparison notebook uses
`distilbert-base-uncased-finetuned-sst-2-english` on a 1,000-review sample
and reports 85% accuracy with weighted F1-score of 0.87.

> **Logistic Regression selected as production model** because it is fast,
> lightweight, interpretable, and already saved for inference through the
> Python helper, FastAPI backend, and Streamlit app.

---

## Key Findings

### Top words driving positive sentiment
excellent, love, perfect, great, fresh, delicious, best, wonderful,
amazing, highly, recommend, fantastic, quality, pleased, satisfied

### Top words driving negative sentiment
terrible, awful, disappointed, waste, return, horrible, disgusting,
never, unfortunately, bad, worse, bland, stale, rancid, useless

### Error analysis insights

The model struggles most with:
1. **Sarcasm** â€” "Oh great, another broken product" (classified as positive)
2. **Conditional praise** â€” "Not bad for the price" (ambiguous)
3. **Short reviews** â€” "Ok." or "Fine." (insufficient signal)

These are known limitations of bag-of-words approaches. The DistilBERT
notebook explores a transformer baseline, but the saved production model
remains the TF-IDF + Logistic Regression pipeline.

### Business impact

At 100,000 reviews/month, the Logistic Regression model can:
- Auto-tag ~91,000 reviews correctly based on test accuracy
- Flag ~9,000 reviews for manual review or confidence-based checks
- Process the entire batch in under 2 minutes
- Reduce manual review time by an estimated 85%

---

## Interactive Prediction

Test the model on any custom review text:

```python
predict("This coffee is absolutely amazing, best I have ever had!")
# Positive (confidence: 96%)

predict("Terrible product, broke after one use. Complete waste of money.")
# Negative (confidence: 99%)

predict("It is okay, nothing special but does the job.")
# Negative (confidence: 75%)
```

---

## How to Run

```bash
git clone https://github.com/Sumant40/NLP-Sentiment-Analysis.git
cd nlp-sentiment-analysis
pip install -r requirements.txt

# Download NLTK stopwords (first time only)
python -c "import nltk; nltk.download('stopwords')"

jupyter notebook
# Run notebooks in order: 01 â†’ 02 â†’ 03 â†’ 04
```

---

## Requirements

```
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
nltk==3.8.1
wordcloud==1.9.2
transformers==4.30.0
torch==2.0.1
jupyter==1.0.0
```

---

## Model Comparison Classical NLP vs Deep Learning

| Factor | TF-IDF + Logistic Regression | DistilBERT |
|--------|-------------------------------|------------|
| Evaluation used | 4,000-review stratified test split | 1,000-review sample |
| Accuracy | 91.0% | 85.0% |
| Weighted F1-score | 0.91 | 0.87 |
| Inference speed | Very fast | Slower on CPU |
| Memory usage | Low | Higher |
| Explainability | High (feature weights) | Low (black box) |
| Project role | Saved production model | Baseline comparison notebook |

---

## Limitations and Future Work

- Training data is from 2002 to 2012 language patterns may have evolved
- Model trained on food reviews â€” may not generalise to other domains
  without fine-tuning
- Does not handle multi-language reviews
- Future work: fine-tune DistilBERT on domain-specific data, add aspect-
  based sentiment (e.g. packaging vs taste vs value), deploy as REST API
  using FastAPI

---

## About

**Sumant Jadiyappagoudar**
Bioengineering graduate | Data Science & Computational Biology
[LinkedIn](https://www.linkedin.com/in/sumant-jadiyappagoudar/) |
[GitHub](https://github.com/Sumant40) |
[Email](mailto:sumantjadiyappagoudar@gmail.com)

---

*Part of my data science portfolio.
Other projects: [HR Attrition ML](#) | [SQL + Dashboard](#) | [A/B Testing](#) | [Pharma Analytics](#)*

