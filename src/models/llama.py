import os
import re
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, pipeline
from sklearn.metrics import classification_report, confusion_matrix


class LlamaModel:
    def __init__(
        self, 
        model_name="meta-llama/Llama-2-7b-chat-hf", 
        device_map="auto", 
        use_auth_token=True):
        
        self.model_name = model_name
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            use_auth_token=use_auth_token,
            padding_side='left', 
            #truncation=True,
            #max_length=1024, 
            #padding='max_length',
            )

        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pipeline = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float16,
            device_map=device_map
        )

   
    def generate_predictions(self, dataloader):
        predictions = []
        true_labels = []
        pred_labels = []
        correct = 0
        total = 0
        examples_shown = 0
        
        with torch.no_grad():
            for batch in dataloader:
                for i in range(len(batch["lyrics"])):
                    lyrics = batch["lyrics"][i]
                    true_decade = batch["decade"][i].item()  
                
                
                    prompt = f"""

                    Question: Based on the used language, from which decade are these lyrics:
                    Lyrics:

                    Long, long year I've sat in this place
                    Baby, baby, what's good I've had
                    When you don't know where I wanna go
                    Find a reason love's left me cold
                    
                    Answer: The lyrics are from the decade 1970.


                    Question: Based on the used language, from which decade are these lyrics:
                    Lyrics:
                    \n{lyrics}\n
                    
                    Answer:
                    """

                    sequences = self.pipeline(
                        prompt,
                        do_sample=True,
                        top_k=10,
                        num_return_sequences=1,
                        eos_token_id=self.tokenizer.eos_token_id,
                        max_new_tokens=32,  
                        temperature=0.7,    
                        min_length=4,       
                    )
                    
                    response = sequences[0]['generated_text']
                
                    try:
                        answer_part = response.split("Answer:")[-1].strip()
                        numbers = ''.join(filter(str.isdigit, answer_part))
                        predicted_decade = int(numbers[:4]) if len(numbers) >= 4 else "Unknown"
                    except (IndexError, ValueError):
                        predicted_decade = "Unknown"
                    
                    total += 1
                    if predicted_decade == true_decade:
                        correct += 1

                    true_labels.append(true_decade)
                    pred_labels.append(predicted_decade)
                    
                    if examples_shown < 100:
                        print("\n" + "="*50)
                        print(f"Example {examples_shown + 1}:")
                        print(f"Model output: {response}")
                        print(f"Predicted Decade: {predicted_decade}")
                        print(f"True Decade: {true_decade}")
                        print(f"Correct: {predicted_decade == true_decade}")
                        examples_shown += 1
                    
                    predictions.append({
                        "lyrics": lyrics,
                        "true_decade": true_decade,
                        "predicted_decade": predicted_decade,
                        "correct": predicted_decade == true_decade
                    })

        accuracy = correct / total if total > 0 else 0
        print(f"\nOverall Accuracy: {accuracy:.2%}")

        self._generate_confusion_matrix(true_labels, pred_labels, save_dir = "/work/NLP/reports/figures" )
        self._generate_classification_report(true_labels, pred_labels, save_dir =  "/work/NLP/reports/figures")

        return predictions, accuracy

    def _generate_confusion_matrix(self, true_labels, pred_labels, save_dir):
        # Convert all labels to strings
        true_labels_str = [str(label) for label in true_labels]
        pred_labels_str = [str(label) for label in pred_labels]

        # Generate the confusion matrix
        cm = confusion_matrix(true_labels_str, pred_labels_str, labels=sorted(set(true_labels_str)))
        labels = sorted(set(true_labels_str))

        # Plot the confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted Decade")
        plt.ylabel("True Decade")
        plt.title("Confusion Matrix")
            
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "confmatrix_llama_unprocessed.png"))
        print(f"Confusion matrix saved to {os.path.join(save_dir, 'confmatrix_llama_unprocessed.png')}")
        plt.close()


    def _generate_classification_report(self, true_labels, pred_labels, save_dir):
        true_labels_str = [str(label) for label in true_labels]
        pred_labels_str = [str(label) for label in pred_labels]

        report = classification_report(true_labels_str, pred_labels_str, labels=sorted(set(true_labels_str)), output_dict=True)

        plt.figure(figsize=(12, 10))
        report_df = pd.DataFrame(report).transpose()
        sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="YlGnBu", fmt=".2f")
        plt.title("Classification Report")
        plt.ylabel("Metrics")
        plt.xlabel("Decades")

        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "classreport_llama_unprocessed.png"))
        print(f"Classification report saved to {os.path.join(save_dir, 'classreport_llama_unprocessed.png')}")
        plt.close()




