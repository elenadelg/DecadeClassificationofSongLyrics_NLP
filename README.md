# Decade Classification of Song Lyrics Using Traditional and Advanced ML Models 🎶
This project applies **Natural Language Processing (NLP)** to predict a song’s release decade from its lyrics. We compare several Machine Learning and Deep Learning models and also explore the capabilities of a Large Language Model (**LlaMA**) for this task.

---

## 1. Introduction 👋
Song lyrics offer a lens into cultural and linguistic shifts across time. Our goal is to determine **which decade** a song is from, purely by analyzing its lyrics. It explores:
- **Language Evolution**: Themes and vocabulary changes  
- **Model Capabilities**: Comparing traditional ML vs. advanced models  
- **LLM Exploration**: Testing whether a large language model trained broadly can outperform models specifically fine-tuned for this classification

---

## 2. Objectives 🎯
1. **Decade Prediction**: Classify songs into the correct decade (1950s–2020s)  
2. **Model Comparison**: Benchmark traditional ML, deep learning, and an LLM  
3. **NLP Insights**: Examine shifting sentiments, themes, and word usage over time

---

## 3. Methodology ⚙️

1. **Data Collection**  
   - A subset of the **5 Million Song Dataset**, filtered to 40,000 English songs (5,000 per decade from 1950s–2020s).  
   - Ensured balanced decade representation.

2. **Data Cleaning & Preprocessing**  
   - Removed non-English lyrics and incomplete entries.  
   - **Key observation**: Training on **raw** (unprocessed) lyrics provided better results than using preprocessed text.

3. **Exploratory Analysis**  
   - **Word Clouds**, **Sentiment Analysis**, and **Topic Modeling** revealed evolving language and themes.

4. **Train/Test Split**  
   - **70%** Training, **10%** Validation, **20%** Testing.

---

## 4. Models 🤖

1. **Logistic Regression (Baseline)**  
   - Straightforward, low computational cost.  
   - ~39% accuracy on raw lyrics.

2. **Random Forest**  
   - Ensemble of decision trees (bagging).  
   - ~37% accuracy; prone to overfitting.

3. **LSTM** (Long Short-Term Memory)  
   - Recurrent neural network suited for sequence data.  
   - Used **GloVe** embeddings.  
   - ~33% accuracy, some difficulty with intermediate decades.

4. **BERT** (Bidirectional Encoder Representations from Transformers)  
   - Pretrained Transformer model, fine-tuned on a smaller subset.  
   - ~38% accuracy; excels at subtle decade distinctions.

5. **LlaMA** (Large Language Model by Meta)  
   - Tested via few-shot prompting only.  
   - ~25% accuracy on a small test set.  
   - Demonstrates large models need task-specific tuning for optimal performance.

---

## 5. Results 📊
- **Logistic Regression** (~39%) and **BERT** (~38%) led in accuracy.  
- **BERT** showed stronger nuance detection, fewer big decade-confusion errors.  
- **Random Forest** and **LSTM** often confused adjacent decades (e.g., 1980s vs. 1990s).  
- **LlaMA** underperformed without specialized training.

---

## 6. Conclusion ✅
- Classifying decades from lyrics alone remains **challenging**—all models reach roughly **3x** random-chance accuracy.  
- **BERT** is best at capturing subtle language changes over decades, but at higher computational cost.  
- **Logistic Regression** is simpler and still highly competitive.  
- Large Language Models (like **LlaMA**) need **fine-tuning** to excel at specialized tasks.

---

## Repository Structure 📁

