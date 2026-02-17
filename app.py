import os
import numpy as np
import pandas as pd
import streamlit as st
import textstat

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

DATA_PATH = os.path.join("data", "clear_corpus.csv")
TEXT_COLUMN = "Excerpt"
TARGET_COLUMN = "Flesch-Reading-Ease"


def score_to_band(score: float):
    if score >= 70:
        return "Easy (approx. Grade 3–5)", "✅ Suitable for lower grades"
    elif score >= 50:
        return "Medium (approx. Grade 6–8)", "⚠️ Moderate difficulty"
    else:
        return "Hard (approx. Grade 9–12)", "❗ Challenging text"


@st.cache_resource
def load_model():
    """Train model in-app (no joblib file required). Cached so bar-bar train nahi hoga."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = df[[TEXT_COLUMN, TARGET_COLUMN]].dropna()

    if len(df) > 5000:
        df = df.sample(5000, random_state=42)

    X = df[TEXT_COLUMN].astype(str)
    y = df[TARGET_COLUMN].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=5,
    )
    model = Ridge(alpha=1.0, random_state=42)
    pipe = Pipeline([
        ("tfidf", tfidf),
        ("regressor", model),
    ])

    pipe.fit(X_train, y_train)

    # optional: console me metrics print ho jayenge
    from sklearn.metrics import mean_squared_error, r2_score
    y_pred = pipe.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"[MODEL] Validation RMSE: {rmse:.3f}")
    print(f"[MODEL] Validation R²:   {r2:.3f}")

    return pipe


def compute_readability_metrics(text: str) -> dict:
    try:
        flesch = textstat.flesch_reading_ease(text)
    except Exception:
        flesch = None
    try:
        fk_grade = textstat.flesch_kincaid_grade(text)
    except Exception:
        fk_grade = None
    try:
        smog = textstat.smog_index(text)
    except Exception:
        smog = None
    try:
        dale_chall = textstat.dale_chall_readability_score(text)
    except Exception:
        dale_chall = None

    return {
        "flesch_reading_ease": flesch,
        "flesch_kincaid_grade": fk_grade,
        "smog_index": smog,
        "dale_chall": dale_chall,
    }


def basic_text_stats(text: str) -> dict:
    words = text.split()
    word_count = len(words)
    sentence_count = max(text.count(".") + text.count("!") + text.count("?"), 1)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    avg_sent_len = word_count / sentence_count if sentence_count > 0 else 0

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": avg_word_len,
        "avg_sentence_length": avg_sent_len,
    }


st.set_page_config(
    page_title="Reading Level Assessment for School Content",
    page_icon="📚",
    layout="wide",
)

with st.sidebar:
    st.title("📚 Reading Level NLP")
    st.markdown(
        """
        **Project:** Reading Level Assessment for School Content  
        **Tech:** NLP (TF-IDF + Ridge Regression), classical readability formulas.  

        **Dataset:**  
        - CommonLit / CLEAR-style corpus  
        - ≈5k excerpts, Grades 3–12, teacher-annotated readability scores.
        """
    )
    st.markdown("---")
    st.markdown("👨‍💻 **How to run (for report):**")
    st.code(
        "pip install -r requirements.txt\n"
        "streamlit run app.py",
        language="bash",
    )
    st.markdown("---")
    st.caption("Created by Amrendra – NLP course project 😉")

st.title("Reading Level Assessment for School Content")
st.markdown(
    """
    Enter any English passage (e.g., from a textbook, worksheet, or story).  
    The model will estimate **how easy or difficult** it is for school students to read.
    """
)

col_input, col_info = st.columns([2, 1])

with col_input:
    default_text = (
        "The sun rose over the quiet village, casting a warm golden light on the "
        "small houses and narrow streets. Children hurried to school with bags "
        "bouncing on their backs, while shopkeepers opened their doors to begin "
        "another busy day."
    )
    text = st.text_area(
        "✏️ Paste your school content here:",
        value=default_text,
        height=250,
    )

    uploaded = st.file_uploader(
        "Or upload a .txt file with your passage",
        type=["txt"],
        help="File content will replace the text box above.",
    )
    if uploaded is not None:
        file_text = uploaded.read().decode("utf-8", errors="ignore")
        text = file_text

    analyze_btn = st.button("🔍 Analyze Reading Level", type="primary")

with col_info:
    st.subheader("About this tool")
    st.markdown(
        """
        - Uses **research-grade dataset**  
        - Predicts a **continuous readability score**  
        - Groups text into **Easy / Medium / Hard**  
        - Shows popular readability indices.
        """
    )

    st.markdown("### Use-cases")
    st.markdown(
        """
        - Match passages to **grade level**  
        - Check if content is **too hard** for target class  
        - Compare different versions of the same text.
        """
    )

st.markdown("---")

if analyze_btn:
    if not text or len(text.strip()) < 30:
        st.error("Please enter at least 1–2 meaningful sentences.")
    else:
        with st.spinner("Training / loading model and running readability analysis..."):
            model = load_model()
            model_score = float(model.predict([text])[0])
            band_label, band_desc = score_to_band(model_score)

            metrics = compute_readability_metrics(text)
            stats = basic_text_stats(text)

        st.subheader("Overall Reading Level")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                label="Model Readability Score (0–100)",
                value=f"{model_score:.1f}",
            )
        with c2:
            st.metric(
                label="Difficulty Band",
                value=band_label,
                delta=band_desc,
            )
        with c3:
            st.metric(
                label="Estimated Target Grades",
                value=band_label.split("(")[-1].replace(")", ""),
            )

        st.markdown("### 🔍 Detailed Analysis")
        left, right = st.columns(2)

        with left:
            st.markdown("#### Classical Readability Metrics")
            metr_table = {
                "Metric": [
                    "Flesch Reading Ease",
                    "Flesch-Kincaid Grade",
                    "SMOG Index",
                    "Dale-Chall Score",
                ],
                "Value": [
                    f"{metrics['flesch_reading_ease']:.2f}" if metrics["flesch_reading_ease"] is not None else "N/A",
                    f"{metrics['flesch_kincaid_grade']:.2f}" if metrics["flesch_kincaid_grade"] is not None else "N/A",
                    f"{metrics['smog_index']:.2f}" if metrics["smog_index"] is not None else "N/A",
                    f"{metrics['dale_chall']:.2f}" if metrics["dale_chall"] is not None else "N/A",
                ],
            }
            st.table(pd.DataFrame(metr_table))

        with right:
            st.markdown("#### Basic Text Statistics")
            stats_table = {
                "Feature": [
                    "Word count",
                    "Sentence count",
                    "Avg. word length",
                    "Avg. sentence length (words)",
                ],
                "Value": [
                    stats["word_count"],
                    stats["sentence_count"],
                    f"{stats['avg_word_length']:.2f}",
                    f"{stats['avg_sentence_length']:.2f}",
                ],
            }
            st.table(pd.DataFrame(stats_table))

        st.markdown("### ✂️ Highlighted Passage")
        st.write(text)
else:
    st.info("👆 Paste text or upload a file, then click **Analyze Reading Level**.")
