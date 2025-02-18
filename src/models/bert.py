import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig


class BertClassifier:
    """
    A wrapper class of BERT for initialization and configuration
    """
    def __init__(
        self,
        num_labels=8,
        hidden_dropout_prob=0.5,
        attention_probs_dropout_prob=0.5,
        lr=1e-5,
        weight_decay=0.05,
        model_name="bert-base-uncased", 
        freeze_layers=7
    ):
        self.config = BertConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob
        )
        self.model = BertForSequenceClassification.from_pretrained(model_name, config=self.config)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            
        # Freeze embeddings
        for param in self.model.bert.embeddings.parameters():
            param.requires_grad = False
        # Freeze the lower encoder layers
        for layer in self.model.bert.encoder.layer[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
