# app.py
# Streamlit app — Train & demo Tweet Sentiment Classifier from uploaded CSV only.
# Usage: streamlit run app.py
# Expects CSV with either:
# - Sentiment140 format (no header): columns -> target,id,date,query,user,text  (target=0/4)
# - Or any CSV with 'text' column and optional 'label' or 'target' column.

import os
import io
import re
import time
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
)

st.set_page_config(page_title="Tweet Sentiment — Upload CSV", layout="wide")
st.title("Tweet Sentiment Classifier — Upload CSV Only")

# -----------------------
# Helpers
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

def detect_and_load_csv(path_or_buffer):
    """Load CSV. Try Sentiment140 headerless format first, else load normally."""
    try:
        # try read small sample to inspect
        sample = pd.read_csv(path_or_buffer, nrows=5, header=None, encoding='latin-1')
        # if first row has 6 columns and first col values like 0/4 it's likely Sentiment140
        if sample.shape[1] >= 6 and sample.iloc[:,0].isin([0,4]).all():
            # rewind buffer if file-like
            if hasattr(path_or_buffer, "seek"):
                path_or_buffer.seek(0)
            cols = ['target','id','date','query','user','text']
            df = pd.read_csv(path_or_buffer, header=None, names=cols, encoding='latin-1')
            return df
    except Exception:
        pass
    # fallback: normal csv read
    if hasattr(path_or_buffer, "seek"):
        path_or_buffer.seek(0)
    df = pd.read_csv(path_or_buffer)
    return df

def prepare_dataframe(df):
    # If Sentiment140 style
    if set(['target','text']).issubset(df.columns):
        df = df[['target','text']].dropna().copy()
        # normalize label
        if df['target'].dtype.kind in 'ifu' or set(df['target'].unique()).issubset({0,4}):
            # map 0->negative,4->positive if present
            df['label'] = df['target'].map({0:'negative',4:'positive'}).fillna(df['target'].astype(str))
        else:
            df['label'] = df['target'].astype(str)
    elif 'text' in df.columns and ('label' in df.columns or 'target' in df.columns):
        # use provided label
        label_col = 'label' if 'label' in df.columns else 'target'
        df = df[['text', label_col]].dropna().copy()
        df = df.rename(columns={label_col: 'label'})
        df['label'] = df['label'].astype(str)
    else:
        raise ValueError("CSV must contain either Sentiment140 format or at least a 'text' column and optional 'label'/'target' column.")
    # clean text
    df['text_clean'] = df['text'].astype(str).apply(clean_text)
    df = df[df['text_clean'].str.strip() != ''].reset_index(drop=True)
    return df

def train_and_evaluate(df, subset_size, test_frac, random_state=42, max_features=10000):
    if subset_size and subset_size < len(df):
        df = df.sample(subset_size, random_state=random_state).reset_index(drop=True)
    X = df['text_clean'].values
    y = df['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, stratify=y, random_state=random_state)
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average='binary', pos_label='positive') if 'positive' in y_test else precision_score(y_test, y_pred, average='macro')),
        "recall": float(recall_score(y_test, y_pred, average='binary', pos_label='positive') if 'positive' in y_test else recall_score(y_test, y_pred, average='macro')),
        "f1": float(f1_score(y_test, y_pred, average='binary', pos_label='positive') if 'positive' in y_test else f1_score(y_test, y_pred, average='macro')),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=np.unique(y)),
        "classes": np.unique(y).tolist()
    }
    # try predict_proba if available
    proba = None
    try:
        proba = clf.predict_proba(X_test_vec)
    except Exception:
        proba = None
    return {"vectorizer": vec, "classifier": clf}, metrics, (X_test, y_test, y_pred, proba)

def plot_confusion(cm, classes):
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_xticks(np.arange(len(classes))); ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(classes))); ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i,j]), ha='center', va='center', color="white" if cm[i,j] > cm.max()/2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    return fig

def plot_roc_pr(y_test, proba, classes):
    # works for binary only (positive class at index 1)
    try:
        if proba is None:
            return None, None
        if proba.shape[1] == 2:
            # map y_test to binary 0/1
            pos_label = classes[1]
            y_bin = np.array([1 if y==pos_label else 0 for y in y_test])
            fpr, tpr, _ = roc_curve(y_bin, proba[:,1])
            roc_auc = auc(fpr, tpr)
            fig1, ax1 = plt.subplots()
            ax1.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            ax1.plot([0,1],[0,1],'--')
            ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR"); ax1.set_title("ROC Curve"); ax1.legend()
            precision, recall, _ = precision_recall_curve(y_bin, proba[:,1])
            ap = average_precision_score(y_bin, proba[:,1])
            fig2, ax2 = plt.subplots()
            ax2.plot(recall, precision, label=f"AP = {ap:.3f}")
            ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.set_title("Precision-Recall"); ax2.legend()
            plt.tight_layout()
            return fig1, fig2
    except Exception:
        pass
    return None, None

# -----------------------
# UI: Upload & config
# -----------------------
st.sidebar.header("Upload CSV & Training Config")
uploaded_file = st.sidebar.file_uploader("Upload CSV (Sentiment140 or CSV with 'text' column)", type=["csv"])
subset_size = st.sidebar.slider("Subset size (for training)", min_value=5000, max_value=200000, value=30000, step=5000)
test_frac = st.sidebar.slider("Test set fraction", min_value=0.05, max_value=0.3, value=0.15, step=0.01)
max_features = st.sidebar.number_input("TF-IDF max features", min_value=1000, max_value=50000, value=10000, step=1000)

st.sidebar.markdown("---")
st.sidebar.write("After uploading CSV click 'Prepare & Train' in main panel.")

# -----------------------
# Main: prepare / train
# -----------------------
st.header("1) Prepare data and train model")
if uploaded_file is None:
    st.info("Upload a CSV file using the left sidebar to begin.")
    st.stop()

# Show preview
try:
    df_raw = detect_and_load_csv(uploaded_file)
    st.subheader("Uploaded file preview")
    st.write(f"Rows: {len(df_raw)}, Columns: {list(df_raw.columns[:10])}")
    st.dataframe(df_raw.head(5))
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

if st.button("Prepare & Train"):
    status = st.empty()
    progress = st.progress(0)

    try:
        status.info("1/5 — Reading uploaded CSV...")
        time.sleep(0.3)

        # ✅ Correct: read CSV into a DataFrame first
        df_raw = detect_and_load_csv(uploaded_file)
        st.write(f"Detected columns: {list(df_raw.columns[:10])}")

        status.info("2/5 — Preparing dataframe (cleaning text, mapping labels)...")
        df = prepare_dataframe(df_raw)
        progress.progress(20)

        status.info(f"3/5 — Training model on {min(subset_size, len(df))} samples...")
        pipeline, metrics, test_data = train_and_evaluate(
            df,
            subset_size=int(subset_size),
            test_frac=float(test_frac),
            max_features=int(max_features)
        )
        progress.progress(80)

        # Save artifacts
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline['vectorizer'], "models/tfidf_vectorizer.joblib")
        joblib.dump(pipeline['classifier'], "models/logreg_classifier.joblib")

        progress.progress(100)
        status.success("✅ Training complete. Models saved to ./models/")

        st.session_state['pipeline'] = pipeline
        st.session_state['metrics'] = metrics
        st.session_state['test_data'] = test_data

        st.subheader("Evaluation (hold-out test)")
        st.write(f"Accuracy: **{metrics['accuracy']:.4f}**")
        st.write(f"Precision: **{metrics['precision']:.4f}**")
        st.write(f"Recall: **{metrics['recall']:.4f}**")
        st.write(f"F1: **{metrics['f1']:.4f}**")

        classes = metrics.get('classes', [])
        st.pyplot(plot_confusion(metrics['confusion_matrix'], classes))

        X_test, y_test, y_pred, proba = test_data
        roc_fig, pr_fig = plot_roc_pr(y_test, proba, classes)
        if roc_fig: st.pyplot(roc_fig)
        if pr_fig: st.pyplot(pr_fig)

    except Exception as e:
        st.error(f"Training failed: {e}")
        progress.progress(0)
        status.error("Training failed.")

# -----------------------
# UI: Single prediction
# -----------------------
st.header("2) Single prediction demo")
if 'pipeline' not in st.session_state:
    st.info("Train the model first (Prepare & Train) or upload another CSV and train.")
else:
    text_in = st.text_area("Enter a tweet to classify", height=120)
    if st.button("Predict single"):
        pipeline = st.session_state['pipeline']
        vec = pipeline['vectorizer']; clf = pipeline['classifier']
        x = clean_text(text_in)
        Xv = vec.transform([x])
        pred = clf.predict(Xv)[0]
        prob = None
        try:
            prob_arr = clf.predict_proba(Xv)[0]
            prob = float(np.max(prob_arr))
        except Exception:
            prob = None
        st.write(f"Prediction: **{pred}**")
        if prob is not None:
            st.write(f"Confidence (max): {prob:.3f}")

# -----------------------
# UI: Batch prediction from CSV
# -----------------------
st.header("3) Batch prediction")
batch_file = st.file_uploader("Upload CSV for batch prediction (must contain 'text' column)", type=["csv"], key="batch_uploader")
if batch_file is not None and 'pipeline' in st.session_state:
    try:
        df_batch = pd.read_csv(batch_file)
        if 'text' not in df_batch.columns:
            st.error("Uploaded CSV must contain a 'text' column.")
        else:
            st.write("Preview:")
            st.dataframe(df_batch.head())
            if st.button("Run batch prediction"):
                pipeline = st.session_state['pipeline']
                vec = pipeline['vectorizer']; clf = pipeline['classifier']
                df_batch['text_clean'] = df_batch['text'].astype(str).apply(clean_text)
                Xv = vec.transform(df_batch['text_clean'])
                preds = clf.predict(Xv)
                df_batch['predicted'] = preds
                try:
                    df_batch['prob_positive'] = clf.predict_proba(Xv)[:,1]
                except Exception:
                    pass
                st.dataframe(df_batch.head(50))
                csv_out = df_batch.to_csv(index=False).encode('utf-8')
                st.download_button("Download predictions CSV", data=csv_out, file_name="predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")

# -----------------------
# UI: Download trained model
# -----------------------
st.header("4) Download trained model artifacts")
if os.path.exists("models/tfidf_vectorizer.joblib") and os.path.exists("models/logreg_classifier.joblib"):
    st.write("Saved artifacts in `./models/`")
    with open("models/tfidf_vectorizer.joblib", "rb") as f:
        vec_bytes = f.read()
    with open("models/logreg_classifier.joblib", "rb") as f:
        clf_bytes = f.read()
    st.download_button("Download TF-IDF vectorizer", data=vec_bytes, file_name="tfidf_vectorizer.joblib", mime="application/octet-stream")
    st.download_button("Download LogisticRegression classifier", data=clf_bytes, file_name="logreg_classifier.joblib", mime="application/octet-stream")
else:
    st.info("No trained artifacts found. Train a model first to enable downloads.")

st.markdown("---")
st.caption("This app uses TF-IDF + Logistic Regression baseline. Upload CSV, train, then predict.")
