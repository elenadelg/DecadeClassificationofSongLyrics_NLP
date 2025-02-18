# **Decade Classification of Song Lyrics Using Traditional and Advanced ML Models**


## 📝 **Introduction**
 This project investigates the linguistic evolution of song lyrics across decades using **Natural Language Processing (NLP)** techniques.
Then, it compares model performances in
predicting a song’s release decade based on its lyrics. 

It explores:
- **Language Evolution:** Tracking thematic and lexical changes across decades.  
- **Model Capabilities:** Comparing traditional machine learning (ML) models with modern language models.  
- **LLM Performance:** Investigating if large language models (LLMs) outperform domain-specific models.

---

## 📊 **Dataset**

A subset of the **Million Song Dataset** ([Bertin-Mahieux et al., 2011][2]) was used, which contains metadata and lyrics for contemporary popular music.  
- Sample size: **40,000** English songs (5,000 per decade from the 1950s to the 2020s).  
- Decades are evenly distributed to avoid class imbalance.

---

## ⚙️ **Methodology**

1. **Data Collection**  
   - Extracted lyrics and metadata from the Million Song Dataset.

2. **Data Cleaning & Preprocessing**  
   - **Cleaning:** Removed non-English lyrics, placeholders, and incomplete records.  
   - **Feature Engineering:** Derived the song's **decade** from its year.  
   - **Text Preprocessing:** Removed punctuation and brackets, filtered noise like *"ooh"* and *"ahh"*, applied lemmatization.

3. **Exploratory Data Analysis (EDA)**  
   - Analyzed trends via **word clouds**, **sentiment analysis**, and **topic modeling**.

4. **Dataset Splitting**  
   - **70%** Training | **10%** Validation | **20%** Testing.  
   - Created both **processed** and **unprocessed** datasets to compare performance.

5. **Model Training**  
   - **Logistic Regression:** Baseline with TF-IDF vectorization.  
   - **Random Forest:** Improved TF-IDF implementation ([Xiang, 2022][1]).  
   - **LSTM:** Recurrent neural network with **GloVe embeddings** ([Pennington et al., 2014][5]).  
   - **BERT:** Pretrained transformer using `bert-base-uncased` ([Devlin et al., 2019][3]).  
   - **LLaMA 2 (7B):** Meta's large language model using 'meta-llama/Llama-2-7b-chat-hf' from Hugging Face ([Meta AI, 2023][4]). It was tested using few-shot prompting. 


---

## 📈 **Results**

| Model          | Accuracy | Macro F1 | Best Decade (F1) | Worst Decade (F1) |
|-----------------|----------|----------|------------------|------------------|
| BERT           | 0.34     | 0.31     | 2010s (0.66)     | 1990s (0.08)     |
| Logistic Reg.  | 0.39     | 0.39     | 2010s (0.60)     | 1990s (0.25)     |
| LSTM           | 0.35     | 0.32     | 2010s (0.64)     | 1990s (0.13)     |
| Random Forest  | 0.37     | 0.34     | 2010s (0.60)     | 1990s (0.21)     |
| LLAMA          | 0.33     | 0.34     | 2010s (0.57)     | 2020s (0.03)     |

- All models perform best with modern lyrics (2010s), suggesting that recent language patterns are more distinct.
- Overall, adjacent decades cause substantial confusion, likely due to gradual linguistic shifts. In particular 1970s, 1980s, and 1990s present significant classification challenges,
- Logistic Regression (~39%) and Random Forest (~37%) led in accuracy.
- LlaMA performed similarly to the other models despite not being fine tuned for the task. 

---

## ✅ **Conclusion**
- Classifying decades from lyrics alone remains challenging—all models reach roughly 3x random-chance accuracy.
- BERT is best at capturing subtle language changes over decades, but at higher computational cost.
- Logistic Regression is simpler and still highly competitive.
- Large Language Models (like LlaMA) need fine-tuning to excel at specialized tasks.

---

## 🔍 **References**

1. Xiang, L. (2022). *Application of an Improved TF-IDF Method in Literary Text Classification*. *Advances in Multimedia*, 2022, Article 9285324. [https://doi.org/10.1155/2022/9285324](https://doi.org/10.1155/2022/9285324)  

2. Bertin-Mahieux, T., Ellis, D. P. W., Whitman, B., & Lamere, P. (2011). *The Million Song Dataset*. In *Proceedings of the 12th International Society for Music Information Retrieval Conference* (ISMIR 2011). [https://labrosa.ee.columbia.edu/millionsong/](https://labrosa.ee.columbia.edu/millionsong/)

3. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. *NAACL-HLT 2019*. [[arXiv:1810.04805](https://arxiv.org/abs/1810.04805)]

4. Meta AI. (2023). *LLaMA: Open and Efficient Foundation Language Models*. *Meta AI Research*. [[GitHub](https://github.com/facebookresearch/llama)]

5. Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global Vectors for Word Representation*. *EMNLP 2014*. [[Paper Link](https://aclanthology.org/D14-1162)]

6. Heimerl, F., Lohmann, S., Lange, S., & Ertl, T. (2014). *Word Cloud Explorer: Text Analytics Based on Word Clouds*. *HICSS*. [[Paper Link](https://ieeexplore.ieee.org/document/6758784)]

7. Pang, B., & Lee, L. (2008). *Opinion Mining and Sentiment Analysis*. *Foundations and Trends in Information Retrieval*, 2(1–2), 1–135. [[Paper Link](https://www.cs.cornell.edu/home/llee/omsa/)]

8. Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). *Latent Dirichlet Allocation*. *Journal of Machine Learning Research*, 3, 993–1022. [[Paper Link](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)]

---

## 🗂️ **Project Structure**

```plaintext
NLP/
   ├── data/
   │    ├── raw/                  # Raw datasets
   │    └── processed/            # Preprocessed datasets
   ├── documents/                 # Project documentation
   ├── myenv/                     # Virtual environment directory
   ├── notebooks/                 # Jupyter notebooks for analysis
   ├── reports/                   # Model performance reports
   ├── src/
   │    ├── data/                 # Data preprocessing and model-specific loaders
   │    └── models/               # Model implementations (BERT, LSTM, etc.)
   ├── .gitignore
   ├── README.md
   ├── requirements.txt
   └── JobParameters.json
```

---

<<<<<<< HEAD
=======

>>>>>>> d626540 (Initial commit for DecadeClassificationofSongLyrics NLP project)
