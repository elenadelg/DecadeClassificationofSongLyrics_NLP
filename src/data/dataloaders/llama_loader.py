import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import logging
from typing import Tuple



class LlamaLoader: 
    def __init__(
        self, 
        tokenizer=None,
        max_length=1024,
        data_path="/work/NLP/data/processed/unprocessed_dataset.csv",
        random_state=42
    ):
        """
        A data-loading and preprocessing class for LLaMA fine-tuning.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_path = data_path
        self.random_state = random_state
        self.textcol = 'text'
        self.lyricscol = 'lyrics'
        self.decadecol = 'decade'

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def decade_mapping (self, df): 
        mapping = {
                0: 1950,
                1: 1960,
                2: 1970,
                3: 1980,
                4: 1990,
                5: 2000,
                6: 2010,
                7: 2020
            }

        df[self.decadecol] = df[self.decadecol].map(mapping)
        if df[self.decadecol].isnull().any():
            raise ValueError("Decade mapping resulted in NaN values. Check the input data.")
        return df


    def create_textcol(self, row: pd.Series, test: bool = False) -> str:
        """
        Modified prompt format for Llama-2-chat
        """

        prompt = f"""<s>[INST]
            Here is an example.
            Question: Based on the used language, please tell me from which decade (from 1950 to 2020) these lyrics are:

            Lyrics:
            Long, long year I've sat in this place
            Baby, baby, what's good I've had
            When you don't know where I wanna go
            Find a reason love's left me cold
            Write the decade of the song and answer with just the decade number. Answer: 1970
            
            Here is the question for you 
            Question: Based on the used language, please tell me from which decade (from 1950 to 2020) these lyrics are:
            Lyrics:
            \n{row[self.lyricscol]}\n
            Write the decade of the song and answer with just the decade number. Answer: [/INST]"""

 
        return prompt



    def tokenize_function(self, example: dict) -> dict:
        """
        Tokenizes the text in self.textcol with truncation and padding.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not provided.")
        
        return self.tokenizer(
            example[self.textcol],
            truncation=True,
            padding='max_length',
            max_length=self.max_length, 
            add_special_tokens=True
        )
    
    def collate_fn(self, batch):
        """
        Custom collate function to properly batch the tokenized examples
        """
        return {
            'input_ids': torch.stack([torch.tensor(example['input_ids']) for example in batch]),
            'attention_mask': torch.stack([torch.tensor(example['attention_mask']) for example in batch]),
            'text': [example['text'] for example in batch],
            'lyrics': [example['lyrics'] for example in batch],
            'decade': torch.tensor([example['decade'] for example in batch])
        }

    def split_dataset(
        self,
        test_size: float = 0.3,
        balance_data: bool = False,
        group_column: str = 'decade'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads a CSV from self.data_path, splits into train/test sets.
        """

        if self.data_path is None:
            raise ValueError("Data path not provided.")
        
        df = pd.read_csv(self.data_path)
        if group_column not in df.columns:
            raise ValueError(f"The column '{group_column}' does not exist in the dataset.")

        df = self.decade_mapping(df)

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df[group_column] if balance_data else None,
            random_state=self.random_state
        )
        self.logger.info("Train/Test Split Summary:")
        self.logger.info("-" * 30)
        self.logger.info(f"Training set size: {len(train_df)}")
        self.logger.info(f"Test set size: {len(test_df)}")
        self.logger.info("Training set class distribution:")
        self.logger.info(train_df[group_column].value_counts().sort_index())
        self.logger.info("\nTest set class distribution:")
        self.logger.info(test_df[group_column].value_counts().sort_index())

        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def createloader(self, train_df, test_df, num_proc=4, batch_size=8
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        self.logger.info("Processing training dataset...")

        train_df[self.textcol] = train_df.apply(lambda row: self.create_textcol(row), axis=1)
        train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
        train_dataset = train_dataset.map(self.tokenize_function, batched=False)
        #train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=self.collate_fn
        )

        test_df[self.textcol] = test_df.apply(lambda row: self.create_textcol(row), axis=1)
        test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))
        test_dataset = test_dataset.map(self.tokenize_function, batched=False)
        #test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=self.collate_fn
        )

        self.logger.info("Training and testing DataLoaders are ready.")
        return train_dataloader, test_dataloader



