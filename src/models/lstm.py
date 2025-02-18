import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, SpatialDropout1D, BatchNormalization, Dropout, Bidirectional, Add
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path


class GloveLSTMModel:
    """
    LSTM model with:
    - Spatial Dropout for embeddings
    - Multiple LSTM layers with residual connections
    - Dropout, Batch Normalization, learning rate scheduling
    """

    def __init__(self, embedding_matrix, max_words=500, max_length=400, num_classes=8, embedding_dim=100):
        self.embedding_matrix = embedding_matrix
        self.max_words = max_words
        self.max_length = max_length
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.model = None

    def build_model(self, dropout_rate=0.3, spatial_dropout=0.2, lstm_units=[128, 96], dense_units=[64],
                   learning_rate=1e-3, trainable=True):

        inputs = Input(shape=(self.max_length,))

        x = Embedding(input_dim=self.max_words,
                    output_dim=self.embedding_dim,
                    embeddings_initializer=tf.keras.initializers.Constant(self.embedding_matrix),
                    input_length=self.max_length,
                    trainable=trainable)(inputs)
        x = SpatialDropout1D(spatial_dropout)(x)

        lstm_out = x
        for i, units in enumerate(lstm_units):
            return_seq = (i != len(lstm_units) - 1)
            lstm_layer = Bidirectional(LSTM(units, return_sequences=return_seq))(lstm_out)
            lstm_layer = BatchNormalization()(lstm_layer)
            lstm_layer = Dropout(dropout_rate)(lstm_layer)
            
            if i != len(lstm_units) - 1:
                if lstm_out.shape[-1] != lstm_layer.shape[-1]:
                    lstm_out = Dense(lstm_layer.shape[-1])(lstm_out)
                lstm_out = Add()([lstm_out, lstm_layer])
            else:
                lstm_out = lstm_layer

        for units in dense_units:
            lstm_out = Dense(units, activation='relu')(lstm_out)
            lstm_out = BatchNormalization()(lstm_out)
            lstm_out = Dropout(dropout_rate)(lstm_out)

        outputs = Dense(self.num_classes, activation='softmax')(lstm_out)
        model = Model(inputs=inputs, outputs=outputs)

        lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(loss='categorical_crossentropy', 
                    optimizer=optimizer,
                    metrics=['accuracy'])

        self.model = model
        return lr_schedule
