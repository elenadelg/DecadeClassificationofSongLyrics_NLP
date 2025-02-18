import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

class ModelEvaluator:
    def __init__(self, y_true, y_pred, target_names, report_type="processed", model_name="model"):
        self.y_true = y_true
        self.y_pred = y_pred
        self.target_names = target_names
        self.report_type = report_type
        self.model_name = model_name

        self.base_dir = "/work/NLP/reports"
        self.figures_dir = os.path.join(self.base_dir, "figures")

        os.makedirs(self.figures_dir, exist_ok=True)

    def save_classification_report(self):
        report = classification_report(self.y_true, self.y_pred, target_names=self.target_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        csv_path = os.path.join(self.base_dir, f"classreport_{self.model_name}_{self.report_type}.csv")
        img_path = os.path.join(self.figures_dir, f"classreport_{self.model_name}_{self.report_type}.png")

        report_df.to_csv(csv_path, index=True)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis("tight")
        ax.axis("off")
        table = ax.table(cellText=report_df.round(2).values, 
                         colLabels=report_df.columns, 
                         rowLabels=report_df.index, 
                         cellLoc="center", 
                         loc="center")

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

        plt.title(f"Classification Report - {self.model_name.upper()} ({self.report_type.capitalize()} Lyrics)")
        plt.savefig(img_path, bbox_inches='tight')
        plt.close()

    def save_confusion_matrix(self):
        labels = np.unique(self.y_true) 

        img_path = os.path.join(self.figures_dir, f"confmatrix_{self.model_name}_{self.report_type}.png")

        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix(self.y_true, self.y_pred, labels=labels), 
                    annot=True, fmt="d", cmap="Blues", xticklabels=self.target_names, yticklabels=self.target_names)

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix - {self.model_name.upper()} ({self.report_type.capitalize()} Lyrics)")

        plt.savefig(img_path, bbox_inches='tight')
        plt.show()

    def evaluate(self):
        self.save_classification_report()
        self.save_confusion_matrix()
