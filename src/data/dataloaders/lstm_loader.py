import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class LSTMDataLoader:
    """
    Prepares data for an LSTM model using tokenization, padding, and GloVe embeddings.
    """
    def __init__(self, glove_file, max_words=500, max_length=400, embedding_dim=100):
        self.max_words = max_words
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.glove_file = glove_file

        # Initialize tokenizer
        self.tokenizer = Tokenizer(num_words=max_words, split=' ', 
                                   filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

        # Embeddings storage
        self.embeddings_index = {}
        self.embedding_matrix = None

    def tokenize_and_pad(self, X_train, X_val, X_test):
        """
        Tokenizes and pads input text sequences for LSTM models.
        Returns padded sequences and tokenizer details.
        """
        X_train, X_val, X_test = self._convert_to_list(X_train, X_val, X_test)
        self.tokenizer.fit_on_texts(X_train)

        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_val_seq   = self.tokenizer.texts_to_sequences(X_val)
        X_test_seq  = self.tokenizer.texts_to_sequences(X_test)

        X_train_padded = pad_sequences(X_train_seq, maxlen=self.max_length)
        X_val_padded   = pad_sequences(X_val_seq,   maxlen=self.max_length)
        X_test_padded  = pad_sequences(X_test_seq,  maxlen=self.max_length)

        self._load_glove_embeddings()
        self._build_embedding_matrix()

        return X_train_padded, X_val_padded, X_test_padded, self.tokenizer.word_index

    def _convert_to_list(self, X_train, X_val, X_test):
        if isinstance(X_train, (pd.DataFrame, pd.Series)):
            X_train = X_train.squeeze().tolist()
        if isinstance(X_val, (pd.DataFrame, pd.Series)):
            X_val = X_val.squeeze().tolist()
        if isinstance(X_test, (pd.DataFrame, pd.Series)):
            X_test = X_test.squeeze().tolist()
        return X_train, X_val, X_test

    def _load_glove_embeddings(self):
        """
        Loads the GloVe embeddings into a dictionary: {word: embedding_vector}.
        """
        with open(self.glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = coefs
        print(f" Loaded {len(self.embeddings_index)} word vectors from GloVe.")

    def _build_embedding_matrix(self):
        """
        Builds an embedding matrix using the tokenizer's word index and the loaded GloVe embeddings.
        """
        self.embedding_matrix = np.zeros((self.max_words, self.embedding_dim))
        for word, i in self.tokenizer.word_index.items():
            if i < self.max_words:
                embedding_vector = self.embeddings_index.get(word)
                if embedding_vector is not None:
                    self.embedding_matrix[i] = embedding_vector
        print("Embedding matrix created.")
