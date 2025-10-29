# app.py

import os
import zipfile
import io
import time
import re
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib

# -----------------------
# Kaggle download helper
# -----------------------
def download_sentiment140_kaggle(target_dir="data"):
    """Download Sentiment140 from Kaggle into target_dir and return csv path."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        raise RuntimeError("kaggle package not found. Install with `pip install kaggle`.") from e

    os.makedirs(target_dir, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    dataset = "kazanova/sentiment140"
    # This downloads a zip file into target_dir
    zip_path = os.path.join(target_dir, "sentiment140.zip")
    api.dataset_download_files(dataset, path=target_dir, unzip=False, quiet=False)
    # Kaggle API creates a zip named like dataset.zip in target_dir
    # Find the zip file produced
    # If API already unzips in some envs, attempt to find csv
    possible_csvs = [
        os.path.join(target_dir, "training.1600000.processed.noemoticon.csv"),
        os.path.join(target_dir, "training.csv"),
        os.path.join(target_dir, "data.csv"),
    ]
    # If a zip exists, unzip it
    zfiles = [f for f in os.listdir(target_dir) if f.endswith(".zip")]
    if zfiles:
        zf = os.path.join(target_dir, zfiles[0])
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(target_dir)
    # search for known csv
    for p in possible_csvs:
        if os.path.exists(p):
            return p
    # fallback: find any csv in dir
    csvs = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(".csv")]
    if csvs:
        return csvs[0]
    raise FileNotFoundError("Could not find CSV after downloading. Check Kaggle dataset contents.")

# -----------------------
# Text cleaning
# -----------------------
def clean_text(s):
    s = "" if pd.isna(s) else str(s)
    s = s.lower()
    s = re.sub(r'http\S+|www\.\S+', '', s)
    s = re.sub(r'@\w+', '', s)
    s = re.sub(r'#', '', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# -----------------------
# Train pipeline
# -----------------------
def train_pipeline(csv_path, subset_size=30000, test_frac=0.15, random_state=42):
    # load with Sentiment140 column names if no header
    cols = ['target','id','date','query','user','text']
    df = pd.read_csv(csv_path, header=None, names=cols, encoding='latin-1')
    df = df[df['target'].isin([0,4])]
    df = df[['target','text']].dropna().copy()
    # map labels
    df['label_text'] = df['target'].map({0:'negative', 4:'positive'})
    # sample for speed
    if subset_size and subset_size < len(df):
        df = df.sample(subset_size, random_state=random_state).reset_index(drop=True)
    # clean
    df['text_clean'] = df['text'].apply(clean_text)
    X = df['text_clean'].values
    y = df['label_text'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, stratify=y, random_state=random_state)
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label='positive')),
        "recall": float(recall_score(y_test, y_pred, pos_label='positive')),
        "f1": float(f1_score(y_test, y_pred, pos_label='positive')),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=['negative','positive'])
    }
    pipeline = {"vectorizer": vec, "classifier": clf}
    return pipeline, metrics, (X_test, y_test, y_pred)

# -----------------------
# Plot helpers
# -----------------------
def plot_confusion(cm, labels=['negative','positive']):
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i,j]), ha='center', va='center',
                    color='white' if cm[i,j] > cm.max()/2. else 'black')
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    return fig

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Tweet Sentiment (Kaggle) Demo", layout="wide")
st.title("Tweet Sentiment Classifier — Train from Kaggle dataset")

st.sidebar.header("Kaggle + Training")
st.sidebar.markdown("Ensure Kaggle credentials present: `~/.kaggle/kaggle.json` or env vars.")
download_button = st.sidebar.button("Download Sentiment140 from Kaggle")
data_dir = st.sidebar.text_input("Data directory", value="data")
subset_size = st.sidebar.number_input("Subset size (for training)", min_value=5000, max_value=1600000, value=30000, step=5000)
test_frac = st.sidebar.slider("Test set fraction", 0.05, 0.3, 0.15)

csv_path = None
if download_button:
    with st.spinner("Downloading dataset from Kaggle..."):
        try:
            csv_path = download_sentiment140_kaggle(target_dir=data_dir)
            st.success(f"Downloaded dataset CSV: {csv_path}")
        except Exception as e:
            st.error(f"Download failed: {e}")
            st.stop()

# If user manually set path
if csv_path is None:
    # check common locations
    candidates = [
        os.path.join(data_dir, "training.1600000.processed.noemoticon.csv"),
        os.path.join(data_dir, "training.csv"),
        os.path.join(data_dir, "data.csv")
    ]
    for c in candidates:
        if os.path.exists(c):
            csv_path = c
            break

if csv_path and os.path.exists(csv_path):
    st.sidebar.success(f"Using CSV: {csv_path}")
    if st.sidebar.button("Train Model Now"):
        with st.spinner("Training model... this may take a while depending on subset size"):
            pipeline, metrics, test_data = train_pipeline(csv_path, subset_size=int(subset_size), test_frac=float(test_frac))
            # save to session
            st.session_state['pipeline'] = pipeline
            st.session_state['metrics'] = metrics
            st.session_state['test_data'] = test_data
            # save files
            os.makedirs("models", exist_ok=True)
            joblib.dump(pipeline['vectorizer'], "models/tfidf_vectorizer.joblib")
            joblib.dump(pipeline['classifier'], "models/logreg_classifier.joblib")
            st.success("Training complete. Models saved to /models/*.joblib")

# Show metrics if available
if 'metrics' in st.session_state:
    st.header("Evaluation Metrics (hold-out test)")
    m = st.session_state['metrics']
    st.write(f"Accuracy: **{m['accuracy']:.4f}**")
    st.write(f"Precision: **{m['precision']:.4f}**")
    st.write(f"Recall: **{m['recall']:.4f}**")
    st.write(f"F1: **{m['f1']:.4f}**")
    st.pyplot(plot_confusion(m['confusion_matrix']))

# Prediction UI
st.header("Prediction")
input_text = st.text_area("Enter a tweet to classify", height=120)
if st.button("Predict") and 'pipeline' in st.session_state:
    vec = st.session_state['pipeline']['vectorizer']
    clf = st.session_state['pipeline']['classifier']
    x = clean_text(input_text)
    Xv = vec.transform([x])
    pred = clf.predict(Xv)[0]
    prob = None
    try:
        prob = clf.predict_proba(Xv)[0].max()
    except Exception:
        pass
    st.write(f"Prediction: **{pred}**")
    if prob is not None:
        st.write(f"Probability (max): {prob:.3f}")

# Batch prediction from uploaded CSV (must have 'text' column)
st.header("Batch Prediction (Upload CSV with 'text' column)")
uploaded = st.file_uploader("Upload CSV", type=['csv'])
if uploaded is not None and 'pipeline' in st.session_state:
    dfu = pd.read_csv(uploaded)
    if 'text' not in dfu.columns:
        st.error("CSV must contain a 'text' column.")
    else:
        dfu['text_clean'] = dfu['text'].astype(str).apply(clean_text)
        vec = st.session_state['pipeline']['vectorizer']
        clf = st.session_state['pipeline']['classifier']
        Xv = vec.transform(dfu['text_clean'])
        preds = clf.predict(Xv)
        dfu['predicted'] = preds
        try:
            dfu['prob_positive'] = clf.predict_proba(Xv)[:,1]
        except Exception:
            pass
        st.dataframe(dfu.head(50))
        csv_out = dfu.to_csv(index=False).encode('utf-8')
        st.download_button("Download predictions", data=csv_out, file_name="predictions.csv", mime="text/csv")

st.markdown("---")
st.caption("This demo trains a TF-IDF + Logistic Regression baseline on Sentiment140 downloaded via Kaggle API.")
