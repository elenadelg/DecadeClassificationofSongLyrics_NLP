import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class RFDataLoader(TfidfVectorizer):  
    ''' 
        Improved TF-IDF vectorization inspired by Lin Xiang's method.
    ''' 
    def __init__(self, max_features=10000, ngram_range=(1,2)):
        super().__init__(max_features=max_features, ngram_range=ngram_range)
        self.doc_freq = None
        self.N = None

    def fit(self, raw_documents, y=None):
        X = super().fit_transform(raw_documents)
        self.doc_freq = np.bincount(X.indices)
        self.N = X.shape[0]
        return self

    def _tfidf(self, X, y=None):
        if self.use_idf:
            df = self.doc_freq
            idf = np.log((self.N - df + 0.5) / (df + 0.5)) + 1.0
            return X.multiply(idf)
        else:
            return X

