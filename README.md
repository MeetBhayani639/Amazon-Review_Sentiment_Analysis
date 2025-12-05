# Amazon Reviews Sentiment Analysis
### A Natural Language Processing (NLP) Project

---

## 📌 Project Overview

This project performs **Sentiment Analysis** on Amazon product reviews.  
The goal is to automatically analyze the text written by users and **predict whether a review is Positive, Negative, or Neutral**. This helps understand customer opinions at scale, improves product analytics, and supports decision-making for e-commerce platforms.

The project is implemented entirely in **Jupyter Notebook**, using Python and NLP libraries such as **NumPy, Pandas, Scikit-Learn, and Transformers (BERT)**.

---

## 🎯 What This Project Does

### ✔️ 1. Loads Amazon Review Data  
The dataset contains Amazon customer reviews with fields such as:
- Review text  
- Rating (1–5 stars)  
- Product category  
- Helpful votes  

---

### ✔️ 2. Converts Ratings into Sentiment Labels  
To train the model, ratings are mapped into sentiment classes:
- **1–2 stars → Negative**  
- **3 stars → Neutral**  
- **4–5 stars → Positive**

---

### ✔️ 3. Cleans & Preprocesses Text  
Standard NLP preprocessing steps include:
- Lowercasing  
- Removing punctuation, symbols, and HTML  
- Tokenizing text into words  
- *(Optional)* Stopword removal & lemmatization  

---

### ✔️ 4. Extracts Features  
Two feature-extraction approaches are used:
- **TF-IDF Vectorization** for traditional ML models  
- **BERT Tokenization** for Transformer-based models  

---

### ✔️ 5. Trains Sentiment Classification Models  

#### 🔹 Traditional Machine Learning
- Logistic Regression  
- Naive Bayes  
- Linear SVM  

#### 🔹 Deep Learning (Optional Extension)
- LSTM / BiLSTM  
- CNN for text  

#### 🔹 Transformer Model
- Fine-tuned **BERT** for sentiment classification  

---

### ✔️ 6. Evaluates Model Performance  
The models are evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

This comparison shows how performance improves from **TF-IDF → LSTM → BERT**.

---

### ✔️ 7. Predicts Sentiment for New Reviews  
A simple function allows users to input a review and get:
- Predicted sentiment  
- Model confidence score  

**Example:**  
> “The camera quality is amazing” → **Positive (97%)**

---

## 🧠 How the Project Works (High-Level Flow)

Load Dataset
↓
Clean & Preprocess Reviews
↓
Convert Ratings → Sentiment Labels
↓
Split into Training & Test Sets
↓
Feature Extraction (TF-IDF or BERT Tokens)
↓
Train ML / Deep Learning / BERT Models
↓
Evaluate Performance
↓
Predict Sentiment for New Inputs
