"""
app.py - Churn Predictor (Upload -> EDA -> Train -> Predict -> SQL Playground)
Features:
- Upload CSV, click Submit to load dataset into session state
- EDA adapts automatically to uploaded dataset
- Train models on click; show metrics, confusion matrix, ROC curve
- Save best-trained model and feature columns in session state
- Single-record prediction UI aligned to training columns
- SQLite in-memory playground (created with check_same_thread=False)
- Professional light styling + Lottie animations
"""
import os
import io
import sqlite3
import json
import requests
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
from streamlit_lottie import st_lottie

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report,
)

# ----------------------------
# Page config & lightweight styling (light professional theme)
# ----------------------------
st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="wide")
CSS = """
<style>
/* Light professional background */
[data-testid="stAppViewContainer"] { background: linear-gradient(180deg,#f5f7fb 0%, #ffffff 100%); }
h1, h2, h3 { color: #0b2545; }
.card {
  background: #ffffff;
  border-radius: 12px;
  padding: 12px;
  box-shadow: 0 6px 18px rgba(17,24,39,0.08);
}
.small-muted { color: #556987; font-size: 13px; }
.stButton>button { background: linear-gradient(90deg,#1f78ff,#3fb0ff); color: white; border-radius: 8px; padding: 8px 10px; font-weight: 700; }
.metric-card { display:flex; gap:12px; }
.metric-box { background:#f8fafc; border-radius:8px; padding:10px; flex:1; text-align:center; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------
# Lottie helper
# ----------------------------
def load_lottie_url(url: str, timeout=6):
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

LOTTIE_TOP = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_tfb3estd.json")  # friendly data animation
LOTTIE_TRAIN = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json")  # training animation

# ----------------------------
# Session state init
# ----------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "conn" not in st.session_state:
    st.session_state.conn = None
if "best_model" not in st.session_state:
    st.session_state.best_model = None
if "feature_columns" not in st.session_state:
    st.session_state.feature_columns = None
if "trained_metrics" not in st.session_state:
    st.session_state.trained_metrics = None

# ----------------------------
# Layout header
# ----------------------------
top_left, top_right = st.columns([3,1])
with top_left:
    st.title("📊 Churn Predictor — Upload • Train • Predict")
    st.markdown("<div class='small-muted'>Upload any CSV, explore dataset, train models, get metrics, predict, and run SQL queries.</div>", unsafe_allow_html=True)
with top_right:
    if LOTTIE_TOP:
        st_lottie(LOTTIE_TOP, height=110)

st.divider()

# ----------------------------
# Utility functions
# ----------------------------
def safe_read_csv(uploaded_file):
    # returns DataFrame or raises
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)

def prepare_sql_memory(df):
    # create an in-memory sqlite3 connection safe for multi-thread use
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    df.to_sql("customers", conn, index=False, if_exists="replace")
    return conn

def compute_metrics(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else None
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}

def plot_confusion(cm, title="Confusion Matrix"):
    z = cm
    x = ["Pred 0","Pred 1"]
    y = ["Actual 0","Actual 1"]
    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorscale="Blues", showscale=True, hoverongaps=False))
    fig.update_layout(title=title, xaxis_title="Predicted", yaxis_title="Actual")
    return fig

def plot_roc(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {roc_auc:.3f}"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
    fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", template="simple_white")
    return fig

# ----------------------------
# Sidebar: Upload + controls
# ----------------------------
st.sidebar.header("Dataset & Controls")
uploaded = st.sidebar.file_uploader("Upload CSV dataset (or leave blank to use sample)", type=["csv", "xlsx"])

use_sample = st.sidebar.checkbox("Use included sample (if no upload)", value=False)

if uploaded is None and not use_sample:
    st.sidebar.info("Upload a CSV (or check 'Use included sample') and press Submit inside the main Upload panel.")
# ----------------------------
# Main tabs
# ----------------------------
tab_upload, tab_eda, tab_train, tab_predict, tab_sql = st.tabs(["Upload", "EDA", "Train", "Predict", "SQL Playground"])

# ----------------------------
# Upload tab
# ----------------------------
with tab_upload:
    st.header("Upload dataset")
    st.markdown("Upload your CSV (or check 'Use included sample' in sidebar). After uploading, click **Submit** to load the dataset and initialize the SQL playground.")
    col1, col2 = st.columns([3,1])
    with col1:
        uploaded_file_widget = uploaded  # use the sidebar uploader
    with col2:
        submit_button = st.button("Submit dataset")

    if submit_button:
        try:
            if uploaded is not None:
                df = safe_read_csv(uploaded)
            elif use_sample:
                # simple built-in sample (Telco-like) generated small dataset if user didn't upload
                sample_path = "tele.csv"
                if os.path.exists(sample_path):
                    df = pd.read_csv(sample_path)
                else:
                    st.error("No sample file found in project folder. Please upload your dataset.")
                    st.stop()
            else:
                st.warning("No dataset uploaded and 'Use included sample' not checked.")
                st.stop()
            if "Churn" in df.columns:
                df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})

            # Basic cleanup: coerce TotalCharges if present
            if "TotalCharges" in df.columns:
                df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
                df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

            st.session_state.df = df.copy()
            st.success(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} cols")

            # setup sqlite in-memory db (thread-safe connection)
            st.session_state.conn = prepare_sql_memory(df)
            st.info("SQLite in-memory DB created and table 'customers' loaded.")
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")

# ----------------------------
# EDA tab
# ----------------------------
with tab_eda:
    st.header("Exploratory Data Analysis")
    if st.session_state.df is None:
        st.warning("Please upload and submit a dataset in the Upload tab first.")
    else:
        df = st.session_state.df
        st.subheader("Preview & Basic Info")
        st.dataframe(df.head(10), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Rows", f"{df.shape[0]:,}")
        with c2:
            st.metric("Columns", f"{df.shape[1]}")
        with c3:
            st.metric("Missing cells", f"{df.isna().sum().sum():,}")

        st.markdown("---")
        st.subheader("Automatic Charts")

        # if there is a binary target named Churn or churn-like, show distribution
        target_candidates = [c for c in df.columns if c.lower() in ["churn", "exited", "label"]]
        if len(target_candidates) > 0:
            target = target_candidates[0]
        else:
            target = None

        if target is not None:
            st.markdown(f"**Detected target column:** `{target}`")
            try:
                counts = df[target].value_counts()
                fig = px.pie(values=counts.values, names=counts.index.astype(str), title=f"{target} distribution", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info("Could not draw target distribution.")

        # Numeric columns - default top 6 correlations with target (if exists)
        numeric = df.select_dtypes(include=np.number)
        if not numeric.empty:
            st.markdown("**Numeric summary**")
            st.dataframe(numeric.describe().T, use_container_width=True)

            # correlation heatmap (small, interactive)
            if numeric.shape[1] <= 30:
                corr = numeric.corr()
                fig = px.imshow(corr, text_auto=True, title="Correlation matrix (numeric)", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Too many numeric columns to show correlation heatmap.")

        # Categorical quick counts for top 4 categorical columns
        cat = df.select_dtypes(exclude=np.number).columns.tolist()
        if len(cat) > 0:
            st.markdown("**Top categorical distributions**")
            for c in cat[:4]:
                vc = df[c].value_counts().nlargest(10)
                fig = px.bar(x=vc.index.astype(str), y=vc.values, title=f"{c} (top categories)", labels={"x": c, "y":"count"}, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Train tab
# ----------------------------
with tab_train:
    st.header("Train models and show metrics")
    if st.session_state.df is None:
        st.warning("Please upload dataset first.")
    else:
        df = st.session_state.df
        # require a binary target 'Churn' or similar
        target_candidates = [c for c in df.columns if c.lower() in ["churn", "exited", "label"]]
        if not target_candidates:
            st.error("No suitable target column detected (expected 'Churn' or similar). Please ensure your dataset has a binary target column.")
        else:
            target = target_candidates[0]
            st.markdown(f"Using target column: **{target}**")

            # ✅ FIX: Convert Yes/No → 1/0 for training
            df[target] = df[target].replace({"Yes": 1, "No": 0, "yes": 1, "no": 0})

            # Train controls
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                test_size_pct = st.slider("Test set (%)", 10, 40, 20)
            with col2:
                rf_estimators = st.slider("Random Forest n_estimators", 50, 400, 200)
            with col3:
                random_state = st.number_input("Random seed", value=42, step=1)

            if st.button("Train models now"):
                try:
                    # Prepare X, y
                    X = df.drop(columns=[target])
                    y = df[target]

                    # quick preprocessing: fill NA numeric with median, categorical with mode
                    X_num = X.select_dtypes(include=np.number).columns.tolist()
                    X_cat = X.select_dtypes(exclude=np.number).columns.tolist()

                    X_proc = X.copy()
                    for c in X_num:
                        X_proc[c] = X_proc[c].fillna(X_proc[c].median())
                    for c in X_cat:
                        X_proc[c] = X_proc[c].fillna(X_proc[c].mode().iloc[0] if not X_proc[c].mode().empty else "")

                    # one-hot encode categorical columns
                    X_proc = pd.get_dummies(X_proc, drop_first=True)

                    X_train, X_test, y_train, y_test = train_test_split(
                        X_proc, y, test_size=test_size_pct/100.0, random_state=int(random_state), stratify=y if len(np.unique(y))>1 else None
                    )

                    # candidate models
                    candidates = {
                        "Logistic Regression": LogisticRegression(max_iter=1000),
                        "Random Forest": RandomForestClassifier(n_estimators=int(rf_estimators), random_state=int(random_state)),
                    }
                    if XGB_AVAILABLE:
                        candidates["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=int(random_state))

                    train_results = []
                    best_auc = -1.0
                    best_name = None
                    best_model_obj = None

                    for name, mdl in candidates.items():
                        mdl.fit(X_train, y_train)
                        preds = mdl.predict(X_test)
                        proba = mdl.predict_proba(X_test)[:,1] if hasattr(mdl, "predict_proba") else None
                        metrics = compute_metrics(y_test, preds, y_proba=proba)
                        train_results.append((name, metrics, preds, proba, mdl))

                        # use ROC AUC as selection if available else accuracy
                        if metrics["roc_auc"] is not None:
                            score = metrics["roc_auc"]
                        else:
                            score = metrics["accuracy"]
                        if score > best_auc:
                            best_auc = score
                            best_name = name
                            best_model_obj = mdl

                    # Save in session state
                    st.session_state.best_model = best_model_obj
                    st.session_state.feature_columns = X_proc.columns.tolist()
                    st.session_state.trained_metrics = {
                        "results": train_results,
                        "X_test": X_test,
                        "y_test": y_test
                    }

                    st.success(f"Training finished. Best: {best_name} (score={best_auc:.3f})")

                    # Show leaderboard summary
                    leaderboard = []
                    for name, metrics, _, _, _ in train_results:
                        leaderboard.append({
                            "Model": name,
                            "Accuracy": metrics["accuracy"],
                            "Precision": metrics["precision"],
                            "Recall": metrics["recall"],
                            "F1": metrics["f1"],
                            "ROC AUC": metrics["roc_auc"] if metrics["roc_auc"] is not None else np.nan
                        })
                    lb_df = pd.DataFrame(leaderboard).sort_values("ROC AUC", ascending=False).reset_index(drop=True)
                    st.markdown("### Leaderboard")
                    st.dataframe(lb_df.style.format({"Accuracy":"{:.3f}", "Precision":"{:.3f}", "Recall":"{:.3f}", "F1":"{:.3f}", "ROC AUC":"{:.3f}"}), use_container_width=True)

                    # Show detailed metrics for each model
                    st.markdown("### Detailed metrics for each model")
                    for name, metrics, preds, proba, mdl in train_results:
                        st.markdown(f"**{name}**")
                        cols = st.columns([1,1,1,1])
                        cols[0].metric("Accuracy", f"{metrics['accuracy']:.3f}")
                        cols[1].metric("Precision", f"{metrics['precision']:.3f}")
                        cols[2].metric("Recall", f"{metrics['recall']:.3f}")
                        cols[3].metric("F1", f"{metrics['f1']:.3f}")

                        # classification report
                        st.text(classification_report(st.session_state.trained_metrics["y_test"], preds, zero_division=0))
                        # confusion matrix
                        cm = confusion_matrix(st.session_state.trained_metrics["y_test"], preds)
                        fig_cm = plot_confusion(cm, title=f"{name} — Confusion Matrix")
                        st.plotly_chart(fig_cm, use_container_width=True)

                        # ROC if available
                        if proba is not None:
                            fig_roc = plot_roc(st.session_state.trained_metrics["y_test"], proba)
                            st.plotly_chart(fig_roc, use_container_width=True)

                except Exception as e:
                    st.error(f"Training failed: {e}")

# ----------------------------
# Predict tab
# ----------------------------
with tab_predict:
    st.header("Single-record prediction (uses the model you trained)")
    if st.session_state.df is None:
        st.warning("Please upload dataset first.")
    elif st.session_state.best_model is None:
        st.warning("Please train a model first in the Train tab.")
    else:
        df = st.session_state.df
        model = st.session_state.best_model
        feat_cols = st.session_state.feature_columns

        st.markdown("Fill the features below (we will align to training columns automatically).")
        # Build input widgets from the original df's feature columns (excluding target)
        target_candidates = [c for c in df.columns if c.lower() in ["churn","exited","label"]]
        if target_candidates:
            target_col = target_candidates[0]
            original_X = df.drop(columns=[target_col])
        else:
            original_X = df.copy()

        # Show a compact form
        input_vals = {}
        cols = st.columns(3)
        for i, col in enumerate(original_X.columns):
            if original_X[col].dtype == "object":
                vals = original_X[col].dropna().unique().tolist()
                input_vals[col] = cols[i % 3].selectbox(col, options=vals)
            else:
                mn = float(original_X[col].min())
                mx = float(original_X[col].max())
                mean = float(original_X[col].median() if not np.isnan(original_X[col].median()) else 0.0)
                input_vals[col] = cols[i % 3].number_input(col, value=mean, min_value=mn, max_value=mx)

        if st.button("Predict"):
            try:
                one = pd.DataFrame([input_vals])
                one_proc = pd.get_dummies(one, drop_first=True)

                # ✅ Align with training features safely
                one_aligned = one_proc.reindex(columns=feat_cols, fill_value=0)

                pred = model.predict(one_aligned)[0]
                prob = model.predict_proba(one_aligned)[0,1] if hasattr(model, "predict_proba") else None

                st.success("✅ Prediction complete")
                st.write(f"**Prediction:** {'Churn' if pred==1 else 'No Churn'}")
                if prob is not None:
                    st.write(f"**Probability (Churn):** {prob:.2%}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")


# ----------------------------
# SQL Playground tab
# ----------------------------
with tab_sql:
    st.header("SQL Playground (SQLite in-memory)")
    if st.session_state.df is None or st.session_state.conn is None:
        st.warning("Please upload dataset and submit it first (Upload tab).")
    else:
        st.markdown("Run SQL queries on the table named `customers` (this is the uploaded dataset). Example: `SELECT * FROM customers LIMIT 10;`")
        query = st.text_area("SQL query", value="SELECT * FROM customers LIMIT 10;", height=130)
        if st.button("Run SQL"):
            try:
                # Use the stored connection created during Submit (check_same_thread=False)
                conn = st.session_state.conn
                res = pd.read_sql_query(query, conn)
                st.dataframe(res, use_container_width=True)
                st.success(f"Query returned {len(res)} rows")
            except Exception as e:
                st.error(f"SQL error: {e}")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("Gnanananda Dharmana        contact me at :gnani1744@gmail.com")
