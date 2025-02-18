from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from evaluate import ModelEvaluator


class LSTMTrainer:
    def __init__(self, model, target_names, reports_dir):
        self.model = model.model  
        self.target_names = target_names
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = self.reports_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)

    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=20, early_stopping_patience=5):
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True),
            ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
        ]

        print("Starting LSTM training...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=callbacks
        )

        stats_df = pd.DataFrame({
            'epoch': range(1, len(history.history['loss']) + 1),
            'train_loss': history.history['loss'],
            'train_accuracy': history.history['accuracy'],
            'val_loss': history.history['val_loss'],
            'val_accuracy': history.history['val_accuracy'],
        })
        stats_df.to_csv(self.reports_dir / "trainingstats_lstm.csv", index=False)
        
        plt.figure(figsize=(12, 5))
        plt.plot(stats_df['epoch'], stats_df['train_loss'], label='Training Loss')
        plt.plot(stats_df['epoch'], stats_df['val_loss'], label='Validation Loss')
        plt.plot(stats_df['epoch'], stats_df['train_accuracy'], label='Training Accuracy')
        plt.plot(stats_df['epoch'], stats_df['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Metrics')
        plt.legend()
        plt.grid(True)
        plt.title('Training Metrics Over Epochs')
        plt.savefig(self.figures_dir / "trainingplots_lstm.png")
        plt.close()
        
        return history

    def evaluate(self, X_test_padded, y_test_unprocessed):
        test_preds = self.model.predict(X_test_padded, batch_size=32)
        test_preds = np.argmax(test_preds, axis=1)
        test_labels = np.array(y_test_unprocessed)
        
        evaluator = ModelEvaluator(
            y_true=test_labels,
            y_pred=test_preds,
            target_names=self.target_names,
            report_type="unprocessed",
            model_name="lstm"
        )
        evaluator.evaluate()