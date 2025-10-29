# app.py
# Streamlit app that auto-installs kaggle, validates credentials,
# downloads Sentiment140, trains TF-IDF+LogReg, and exposes demo UI.
#
# Run: streamlit run app.py
# Requirements: Python 3.8+. The app will attempt to pip-install missing packages.

import os
import sys
import subprocess
import time
import zipfile
import re
import io

# ---- Auto-install required packages if missing ----
def ensure_package(pkg_name):
    try:
        __import__(pkg_name)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
        time.sleep(0.5)  # allow environment to settle

for p in ("streamlit", "pandas", "numpy", "scikit-learn", "joblib", "matplotlib", "kaggle"):
    ensure_package(p)

# Now safe to import heavier libs
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import joblib
from kaggle.api.kaggle_api_extended import KaggleApi

# ---- Utilities ----
def has_kaggle_credentials():
    # env-vars override
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True, "env"
    # default file location
    home = os.path.expanduser("~")
    path = os.path.join(home, ".kaggle", "kaggle.json")
    if os.path.exists(path):
        return True, path
    return False, None

def validate_kaggle_access():
    ok, loc = has_kaggle_credentials()
    if not ok:
        return False, "Kaggle credentials not found. Provide ~/.kaggle/kaggle.json or set KAGGLE_USERNAME/KAGGLE_KEY env vars."
    try:
        api = KaggleApi()
        api.authenticate()
        return True, "Authenticated"
    except Exception as e:
        return False, f"Kaggle authentication failed: {e}"

def download_sentiment140_kaggle(target_dir="data"):
    os.makedirs(target_dir, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    dataset = "kazanova/sentiment140"
    # downloads dataset zip into target_dir (may create zip)
    api.dataset_download_files(dataset, path=target_dir, unzip=False, quiet=False)
    # find zip(s) and extract
    zfiles = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(".zip")]
    for zf in zfiles:
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(target_dir)
    # known filenames
    candidates = [
        os.path.join(target_dir, "training.1600000.processed.noemoticon.csv"),
        os.path.join(target_dir, "training.csv"),
        os.path.join(target_dir, "data.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # fallback: any csv
    csvs = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(".csv")]
    if csvs:
        return csvs[0]
    raise FileNotFoundError("CSV not found after Kaggle download. Inspect target directory.")

def clean_text(s):
    s = "" if pd.isna(s) else str(s)
    s = s.lower()
    s = re.sub(r'http\S+|www\.\S+', '', s)
    s = re.sub(r'@\w+', '', s)
    s = re.sub(r'#', '', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def train_pipeline(csv_path, subset_size=30000, test_frac=0.15, random_state=42):
    cols = ['target','id','date','query','user','text']
    df = pd.read_csv(csv_path, header=None, names=cols, encoding='latin-1')
    df = df[df['target'].isin([0,4])]
    df = df[['target','text']].dropna().copy()
    df['label_text'] = df['target'].map({0:'negative', 4:'positive'})
    if subset_size and subset_size < len(df):
        df = df.sample(subset_size, random_state=random_state).reset_index(drop=True)
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

def plot_roc_pr(y_true, y_score):
    fig1, ax1 = plt.subplots()
    fpr, tpr, _ = roc_curve(y_true, y_score[:,1])
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax1.plot([0,1],[0,1],'--')
    ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate"); ax1.set_title("ROC Curve"); ax1.legend()
    fig2, ax2 = plt.subplots()
    precision, recall, _ = precision_recall_curve(y_true, y_score[:,1])
    ap = average_precision_score(y_true, y_score[:,1])
    ax2.plot(recall, precision, label=f"AP = {ap:.3f}")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.set_title("Precision-Recall"); ax2.legend()
    plt.tight_layout()
    return fig1, fig2

# ---- Streamlit UI ----
st.set_page_config(page_title="Tweet Sentiment (Kaggle Auto)", layout="wide")
st.title("Tweet Sentiment Classifier — Kaggle Auto-Install + Credential Check")

st.sidebar.header("Setup & Kaggle")
cred_ok, cred_msg = has_kaggle_credentials()
st.sidebar.write(f"Local kaggle credential: {cred_ok}")
if cred_ok:
    st.sidebar.write(f"Credential location: {cred_msg}")
else:
    st.sidebar.info("No local kaggle.json found. You can set KAGGLE_USERNAME and KAGGLE_KEY env vars or place kaggle.json in ~/.kaggle/")

auth_check = st.sidebar.button("Validate Kaggle Authentication")
if auth_check:
    ok, msg = validate_kaggle_access()
    if ok:
        st.sidebar.success("Kaggle authenticated.")
    else:
        st.sidebar.error(msg)

st.sidebar.markdown("---")
data_dir = st.sidebar.text_input("Data directory", value="data")
download_btn = st.sidebar.button("Download Sentiment140 from Kaggle")

subset_size = st.sidebar.number_input("Subset size (train)", min_value=5000, max_value=1600000, value=30000, step=5000)
test_frac = st.sidebar.slider("Test fraction", 0.05, 0.3, 0.15)

csv_path = None
if download_btn:
    ok, _ = validate_kaggle_access()
    if not ok:
        st.error("Kaggle credentials invalid. Fix credentials and retry.")
    else:
        with st.spinner("Downloading Sentiment140 via Kaggle API..."):
            try:
                csv_path = download_sentiment140_kaggle(target_dir=data_dir)
                st.success(f"Downloaded dataset CSV: {csv_path}")
            except Exception as e:
                st.error(f"Download failed: {e}")
                csv_path = None

# Allow user to optionally point to existing CSV
st.sidebar.markdown("Or specify local CSV path if already downloaded")
local_csv = st.sidebar.text_input("Local CSV path (optional)", value="")
if local_csv:
    if os.path.exists(local_csv):
        csv_path = local_csv
        st.sidebar.success(f"Using local CSV: {csv_path}")
    else:
        st.sidebar.warning("Local CSV path not found.")

# If csv found, allow training
if csv_path and os.path.exists(csv_path):
    st.success(f"Dataset ready: {csv_path}")
    if st.button("Train Model Now"):
        with st.status("🚀 Starting training...", expanded=True) as status:
            try:
                st.write("1️⃣ Loading and cleaning dataset...")
                cols = ['target','id','date','query','user','text']
                df = pd.read_csv(csv_path, header=None, names=cols, encoding='latin-1')
                df = df[df['target'].isin([0,4])]
                df = df[['target','text']].dropna().copy()
                df['label_text'] = df['target'].map({0:'negative', 4:'positive'})
                if subset_size and subset_size < len(df):
                    df = df.sample(int(subset_size), random_state=42).reset_index(drop=True)
                df['text_clean'] = df['text'].astype(str).apply(clean_text)
                st.write(f"✅ Loaded {len(df)} samples")

                progress = st.progress(0)
                st.write("2️⃣ Splitting data...")
                X = df['text_clean'].values
                y = df['label_text'].values
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=float(test_frac), stratify=y, random_state=42
                )
                progress.progress(20)

                st.write("3️⃣ Vectorizing text with TF-IDF...")
                vec = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
                X_train_vec = vec.fit_transform(X_train)
                X_test_vec = vec.transform(X_test)
                progress.progress(50)

                st.write("4️⃣ Training Logistic Regression model...")
                clf = LogisticRegression(max_iter=1000, C=1.0)
                clf.fit(X_train_vec, y_train)
                progress.progress(80)

                st.write("5️⃣ Evaluating model...")
                y_pred = clf.predict(X_test_vec)
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, pos_label='positive')
                rec = recall_score(y_test, y_pred, pos_label='positive')
                f1 = f1_score(y_test, y_pred, pos_label='positive')
                cm = confusion_matrix(y_test, y_pred, labels=['negative','positive'])
                progress.progress(100)

                st.session_state['pipeline'] = {'vectorizer': vec, 'classifier': clf}
                st.session_state['metrics'] = {
                    'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'cm': cm
                }
                os.makedirs("models", exist_ok=True)
                joblib.dump(vec, "models/tfidf_vectorizer.joblib")
                joblib.dump(clf, "models/logreg_classifier.joblib")

                st.success("✅ Training complete!")
                st.write(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
                st.pyplot(plot_confusion(cm))
                status.update(label="🎯 Training completed successfully!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Training failed: {e}")
                status.update(label="❌ Training failed", state="error", expanded=True)


# If user uploaded CSV manually, allow training from upload
st.header("Alternative: Upload CSV and train (smaller subsets recommended)")
uploaded = st.file_uploader("Upload Sentiment CSV (optional)", type=["csv"])
if uploaded is not None:
    try:
        tmp_df = pd.read_csv(uploaded)
        st.write("Uploaded sample:")
        st.dataframe(tmp_df.head())
        if st.button("Train on uploaded CSV"):
            # Save uploaded to temp and train using same pipeline (assume Sentiment140 format or columns target/text)
            tmp_path = "uploaded_sentiment.csv"
            uploaded.seek(0)
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getvalue())
            # try to detect format; if header exists try to use columns
            try:
                pipeline, metrics, test_data = train_pipeline(tmp_path, subset_size=int(subset_size), test_frac=float(test_frac))
                st.session_state['pipeline'] = pipeline
                st.session_state['metrics'] = metrics
                st.session_state['test_data'] = test_data
                os.makedirs("models", exist_ok=True)
                joblib.dump(pipeline['vectorizer'], "models/tfidf_vectorizer.joblib")
                joblib.dump(pipeline['classifier'], "models/logreg_classifier.joblib")
                st.success("Training on uploaded CSV complete.")
            except Exception as e:
                st.error(f"Training on uploaded CSV failed: {e}")
    except Exception as e:
        st.error(f"Could not read uploaded CSV: {e}")

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
st.header("Prediction Demo")
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

# Batch prediction for user-uploaded csv with 'text' column
st.header("Batch Prediction (CSV with 'text' column)")
batch_uploaded = st.file_uploader("Upload CSV for batch prediction", type=["csv"], key="batch")
if batch_uploaded is not None and 'pipeline' in st.session_state:
    try:
        dfu = pd.read_csv(batch_uploaded)
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
            st.download_button("Download predictions CSV", data=csv_out, file_name="predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")

st.markdown("---")
