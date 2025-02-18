from sklearn.feature_extraction.text import TfidfVectorizer

class LogRegDataLoader:
    """Prepares data for Logistic Regression using TF-IDF vectorization."""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)

    def vectorize(self, X_train, X_val, X_test):

        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train).toarray()  # Convert sparse matrix to dense
        X_val_tfidf   = self.tfidf_vectorizer.transform(X_val).toarray()
        X_test_tfidf  = self.tfidf_vectorizer.transform(X_test).toarray()

        return X_train_tfidf, X_val_tfidf, X_test_tfidf