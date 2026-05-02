from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf():
    return TfidfVectorizer(
        max_features=5000,
        ngram_range=(1,2),
        min_df=5
    )