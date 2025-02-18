import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from evaluate import ModelEvaluator

class BERTTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        num_epochs=3,
        early_stopping_patience=3,
        target_names=['1950s', '1960s', '1970s', '1980s', '1990s', '2000s', '2010s', '2020s']
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.target_names = target_names
        
        self.model.model.to(self.device)
        
    def train_epoch(self):
        self.model.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(self.train_loader, desc="Training"):
            input_ids = batch[0].to(self.device)
            attention_mask = batch[1].to(self.device)
            labels = batch[2].to(self.device)
            
            self.model.optimizer.zero_grad()
            
            outputs = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            self.model.optimizer.step()
            
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_accuracy = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_accuracy, all_preds, all_labels
    
    def evaluate(self, data_loader, phase="val"):
        self.model.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {phase}"):
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                
                outputs = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = total_loss / len(data_loader)
        epoch_accuracy = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_accuracy, all_preds, all_labels
    
    def save_training_statistics(self, history):
        statistics = {
            "train_loss": history['train_loss'],
            "train_accuracy": history['train_acc'],
            "val_loss": history['val_loss'],
            "val_accuracy": history['val_acc']
        }
        stats_df = pd.DataFrame(statistics)
        stats_csv_path = "/work/NLP/reports/trainingstats_bert.csv"
        stats_df.to_csv(stats_csv_path, index=False)
        print(f"Training stats saved to {stats_csv_path}")
        
    def plot_training_history(self, history):
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        
        axs[0].plot(history['train_loss'], label="Train Loss", linestyle="-", marker="o")
        axs[0].plot(history['val_loss'], label="Val Loss", linestyle="--", marker="s")
        axs[0].set_title("Loss over Epochs")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].legend()

        axs[1].plot(history['train_acc'], label="Train Accuracy", linestyle="-", marker="o")
        axs[1].plot(history['val_acc'], label="Val Accuracy", linestyle="--", marker="s")
        axs[1].set_title("Accuracy over Epochs")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Accuracy")
        axs[1].legend()

        fig_path = "/work/NLP/reports/figures/trainingplots_bert.png"
        plt.savefig(fig_path)
        print(f"Training and Validation plots saved to {fig_path}")
        plt.show()
        plt.close(fig)
    
    def train(self):
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 30)
            
            train_loss, train_acc, train_preds, train_labels = self.train_epoch()
            val_loss, val_acc, val_preds, val_labels = self.evaluate(self.val_loader, "val")
            
            training_history['train_loss'].append(train_loss)
            training_history['train_acc'].append(train_acc)
            training_history['val_loss'].append(val_loss)
            training_history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.model.state_dict(), 'best_bert_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print("\nEarly stopping triggered!")
                    break
        
        self.model.model.load_state_dict(torch.load('best_bert_model.pt'))
        
        test_loss, test_acc, test_preds, test_labels = self.evaluate(self.test_loader, "test")
        print("\nFinal Test Results:")
        print(f"Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.4f}")
        
        self.save_training_statistics(training_history)
        self.plot_training_history(training_history)
        
        evaluator = ModelEvaluator(
            y_true=test_labels,
            y_pred=test_preds,
            target_names=self.target_names,
            report_type="unprocessed",
            model_name="bert"
        )
        evaluator.evaluate()
        
        return training_history, (test_preds, test_labels)


