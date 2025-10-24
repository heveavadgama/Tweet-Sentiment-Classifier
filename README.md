# 🧠 Tweet Sentiment Classifier (NLP Project)

A **Natural Language Processing (NLP)** mini-project that classifies tweets as **positive** or **negative** using **TF-IDF** vectorization and a **Logistic Regression** model.  
Built for learning and demonstration of core NLP workflows — preprocessing, feature extraction, modeling, and evaluation.

---

## 📁 Project Overview

**Objective:**  
To classify the sentiment of tweets using classical NLP methods with low computational cost and strong performance.

**Key Features:**
- End-to-end NLP pipeline (cleaning → vectorization → classification → evaluation)
- Trained and tested on the **Sentiment140** dataset (~1.6M tweets)
- Model interpretability with top-word feature analysis
- Visualization of ROC, Precision–Recall curves, and word importance

---

## 📊 Dataset

- **Name:** [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Size:** ~1.6 million tweets (subset used for faster computation)
- **Labels:**  
  - `0` → Negative  
  - `4` → Positive  
- **Preprocessing:**  
  - Lowercasing  
  - Removing URLs, mentions, hashtags  
  - Removing punctuation and symbols  
  - Token normalization and whitespace cleanup

---

## ⚙️ Methodology

| Step | Description |
|:--|:--|
| **Text Vectorization** | TF-IDF (1–2 grams, max 10,000 features) |
| **Model** | Logistic Regression (C = 1.0) |
| **Search** | GridSearchCV (3-fold cross-validation) |
| **Split** | 85% train / 15% test |

---

## 🧪 Results

| Metric | Value |
|:--|:--|
| **Accuracy** | 0.804 |
| **Precision** | 0.797 |
| **Recall** | 0.815 |
| **F1-score** | 0.806 |
| **ROC-AUC** | 0.884 |
| **Matthews Corr. Coef.** | 0.608 |
| **Cohen’s Kappa** | 0.608 |

✅ Balanced precision and recall  
✅ Low overfitting  
✅ Strong AUC indicating clear class separation

---

## 📉 Evaluation Visuals

- ROC Curve (AUC = 0.884)  
- Precision–Recall Curve  
- Confusion Matrix  
- Top Positive and Negative Tokens  

**Top Positive Tokens:**  
`cant wait, not bad, no problem, smile, happy, congratulations, awesome, glad, excited`  

**Top Negative Tokens:**  
`sad, sadly, miss, unfortunately, disappointed, sick, horrible, headache, upset`

---

## 💡 Error Analysis

- Tweets with **mixed sentiment** (e.g., “not bad”, “sadly happy”) often misclassified  
- **Sarcastic negatives** can be incorrectly labeled as positive  
- **Short, context-poor tweets** occasionally misread  

---

## 🧰 Technologies Used

- Python 3.x  
- Scikit-learn  
- Pandas, NumPy  
- Matplotlib / Seaborn  
- Joblib  
- Regex for preprocessing  

---

