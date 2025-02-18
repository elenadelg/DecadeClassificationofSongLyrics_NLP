import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
from transformers import BertTokenizer

class BERTDataLoader:
    def __init__(
        self,
        csv_path="/work/NLP/data/processed/unprocessed_dataset.csv",
        batch_size=16,
        random_state=42,
        bert_model_name="bert-base-uncased",
        max_length=512  # Added max_length parameter
    ):
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.csv_path = csv_path
        self.random_state = random_state


    def split_data(self):
        """Split data with stratification and validation"""
        df = pd.read_csv(self.csv_path)
        lyrics = df["lyrics"].tolist()
        decades = df["decade"].tolist()
        
        lyrics_train, lyrics_temp, decades_train, decades_temp = train_test_split(
            lyrics, decades,
            test_size=0.3,
            random_state=self.random_state,
            stratify=decades
        )
        
        lyrics_val, lyrics_test, decades_val, decades_test = train_test_split(
            lyrics_temp, decades_temp,
            test_size=0.67,  # Approximately 20% of total for test
            random_state=self.random_state,
            stratify=decades_temp
        )
        
        total = len(lyrics)
        expected_ratios = {'train': 0.7, 'val': 0.1, 'test': 0.2}
        actual_ratios = {
            'train': len(lyrics_train) / total,
            'val': len(lyrics_val) / total,
            'test': len(lyrics_test) / total
        }
        
        print("\nData Split Summary:")
        for split_name, ratio in actual_ratios.items():
            print(f"{split_name.capitalize()} set: {ratio:.2%} "
                  f"(Expected: {expected_ratios[split_name]:.2%})")
        
        return lyrics_train, lyrics_val, lyrics_test, decades_train, decades_val, decades_test

    def tokenize_and_create_loaders(self, lyrics_train, lyrics_val, lyrics_test, 
                                  decades_train, decades_val, decades_test):
        
        """Tokenize texts and create data loaders with additional validation"""
        encodings = {}
        for name, texts in [('train', lyrics_train), ('val', lyrics_val), ('test', lyrics_test)]:
            print(f"\nTokenizing {name} set...")
            encodings[name] = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            max_tokens = encodings[name]['input_ids'].shape[1]
            print(f"Max sequence length in {name} set: {max_tokens}")
            if max_tokens == self.max_length:
                print(f"Warning: Some sequences in {name} set were truncated")
        
        datasets = {
            'train': TensorDataset(
                encodings['train']['input_ids'],
                encodings['train']['attention_mask'],
                torch.tensor(decades_train)
            ),
            'val': TensorDataset(
                encodings['val']['input_ids'],
                encodings['val']['attention_mask'],
                torch.tensor(decades_val)
            ),
            'test': TensorDataset(
                encodings['test']['input_ids'],
                encodings['test']['attention_mask'],
                torch.tensor(decades_test)
            )
        }
        
        loaders = {
            'train': TorchDataLoader(datasets['train'], batch_size=self.batch_size, shuffle=True),
            'val': TorchDataLoader(datasets['val'], batch_size=self.batch_size, shuffle=False),
            'test': TorchDataLoader(datasets['test'], batch_size=self.batch_size, shuffle=False)
        }
        
        return loaders['train'], loaders['val'], loaders['test']


