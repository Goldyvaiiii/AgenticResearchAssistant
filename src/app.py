# Main Streamlit UI for ArXiv Research Topic Analysis

import os
import sys
import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from collections import Counter

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_arxiv_sample, load_random_sample
from preprocessor import preprocess_corpus, tokens_to_string, get_sentences
from topic_modeler import TopicModeler
from summarizer import extractive_summarize, summarize_topic_group

# Page Config
st.set_page_config(
    page_title="ArXiv Research Topic Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        color: #e6edf3;
    }

    h1 { color: #58a6ff !important; font-weight: 700; }
    h2 { color: #79c0ff !important; font-weight: 600; }
    h3 { color: #a5d6ff !important; }

    .metric-card {
        background: rgba(33, 38, 45, 0.8);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        backdrop-filter: blur(10px);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }

    .keyword-chip {
        display: inline-block;
        background: rgba(88, 166, 255, 0.15);
        border: 1px solid rgba(88, 166, 255, 0.4);
        color: #79c0ff;
        border-radius: 20px;
        padding: 2px 10px;
        margin: 2px;
        font-size: 0.82rem;
    }

    .summary-box {
        background: rgba(22, 27, 34, 0.9);
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem 1.4rem;
        font-size: 0.92rem;
        line-height: 1.7;
        color: #c9d1d9;
    }

    .stButton > button {
        background: linear-gradient(135deg, #1f6feb, #388bfd);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(56, 139, 253, 0.4);
    }

    section[data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid #30363d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Dataset path — auto-detect full vs sample
_src_dir = os.path.dirname(os.path.abspath(__file__))
_data_dir = os.path.join(_src_dir, "..", "data")
FULL_DATA_PATH   = os.path.join(_data_dir, "arxiv-metadata-oai-snapshot.csv")
SAMPLE_DATA_PATH = os.path.join(_data_dir, "arxiv_sample.csv")

# Decide which dataset to use at startup
if os.path.exists(FULL_DATA_PATH):
    DATA_PATH   = FULL_DATA_PATH
    DATA_MODE   = "full"
elif os.path.exists(SAMPLE_DATA_PATH):
    DATA_PATH   = SAMPLE_DATA_PATH
    DATA_MODE   = "sample"
else:
    DATA_PATH   = None
    DATA_MODE   = "none"

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    if DATA_MODE == "sample":
        st.info("☁️ **Cloud mode** — using pre-sampled dataset (2,000 papers). "
                "Run locally with the full ArXiv dataset for more results.", icon="ℹ️")
    elif DATA_MODE == "none":
        st.error("❌ No dataset found. See README for setup instructions.")

    keyword = st.text_input(
        "🔍 Research Topic / Keywords",
        value="machine learning",
        help="Enter keywords to filter papers (e.g., 'deep learning', 'quantum computing')",
    )

    max_papers = 2000 if DATA_MODE == "sample" else 3000
    sample_size = st.slider(
        "📄 Papers to Analyze",
        min_value=100,
        max_value=max_papers,
        value=min(500, max_papers),
        step=100,
    )

    n_topics = st.slider("📊 Number of Topics", min_value=3, max_value=15, value=7)

    n_top_words = st.slider("🏷️ Keywords per Topic", min_value=5, max_value=20, value=10)

    modeling_method = st.selectbox(
        "🧠 Topic Modeling Method",
        options=["lda", "kmeans"],
        format_func=lambda x: "LDA (Latent Dirichlet Allocation)" if x == "lda" else "K-Means Clustering",
    )

    n_summary_sentences = st.slider("📝 Summary Sentences", min_value=1, max_value=6, value=3)

    st.markdown("---")
    run_analysis = st.button("🚀 Run Analysis", use_container_width=True, disabled=(DATA_MODE == "none"))

    st.markdown("---")
    st.markdown(
        "<small style='color:#8b949e'>**Milestone 1** · Traditional NLP Pipeline<br>"
        "ArXiv Research Assistant · 2026</small>",
        unsafe_allow_html=True,
    )

# Header
st.markdown(
    """
    <div style='text-align:center; padding: 1.5rem 0 0.5rem 0;'>
        <h1 style='font-size:2.4rem; margin-bottom:0'>🔬 ArXiv Research Topic Analyzer</h1>
        <p style='color:#8b949e; font-size:1.05rem; margin-top:0.3rem'>
            Traditional NLP Pipeline &middot; TF-IDF &middot; LDA / K-Means &middot; Extractive Summarization
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")


# Helper: load data depending on dataset mode
def _load_data(keyword: str, sample_size: int) -> pd.DataFrame:
    if DATA_MODE == "full":
        return load_arxiv_sample(DATA_PATH, sample_size=sample_size,
                                  keyword_filter=keyword if keyword.strip() else None)
    else:
        # Load CSV sample directly (already small enough to fit in memory)
        df = pd.read_csv(DATA_PATH)
        df = df.fillna("")
        if keyword.strip():
            kw = keyword.lower()
            mask = df["abstract"].str.lower().str.contains(kw, na=False) | \
                   df["title"].str.lower().str.contains(kw, na=False)
            df = df[mask]
        return df.head(sample_size).reset_index(drop=True)


# Helper: cached analysis pipeline
@st.cache_data(show_spinner=False, ttl=600)
def run_pipeline(keyword: str, sample_size: int, n_topics: int, n_top_words: int,
                 method: str, n_summary_sentences: int, data_mode: str):
    df = _load_data(keyword, sample_size)
    if df.empty:
        return None, None, None, None, None

    abstracts = df["abstract"].fillna("").tolist()
    processed = preprocess_corpus(abstracts, use_spacy=True)
    processed_strings = [tokens_to_string(t) for t in processed]

    valid_mask = [bool(s.strip()) for s in processed_strings]
    df = df[valid_mask].reset_index(drop=True)
    processed_strings = [s for s, v in zip(processed_strings, valid_mask) if v]

    if not processed_strings:
        return None, None, None, None, None

    actual_topics = min(n_topics, max(2, len(processed_strings) // 2))

    modeler = TopicModeler(n_topics=actual_topics, n_top_words=n_top_words, method=method)
    modeler.fit(processed_strings)
    topics = modeler.get_topics()
    df = modeler.assign_topics_to_df(df)
    global_keywords = modeler.get_global_top_keywords(top_n=50)

    topic_summaries = {}
    for topic in topics:
        tid = topic["topic_id"]
        group = df[df["dominant_topic"] == tid]["abstract"].dropna().tolist()
        topic_summaries[tid] = summarize_topic_group(group, n_summary_sentences) if group else ""

    return df, topics, global_keywords, topic_summaries, modeler


# Main logic
if run_analysis:
    with st.spinner("⚙️ Running NLP pipeline — loading data & modeling topics…"):
        t0 = time.time()
        result = run_pipeline(
            keyword=keyword.strip(),
            sample_size=sample_size,
            n_topics=n_topics,
            n_top_words=n_top_words,
            method=modeling_method,
            n_summary_sentences=n_summary_sentences,
            data_mode=DATA_MODE,
        )
        elapsed = time.time() - t0

    df, topics, global_keywords, topic_summaries, modeler = result

    if df is None or df.empty:
        st.error("❌ No papers found for that keyword. Try a broader search term.")
        st.stop()

    # 1. Overview Metrics
    st.markdown("## 📊 Overview")
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("📄", "Papers Loaded", len(df)),
        ("🗂️", "Topics Found", len(topics)),
        ("🔑", "Unique Keywords", len(global_keywords)),
        ("⏱️", "Analysis Time", f"{elapsed:.1f}s"),
    ]
    for col, (icon, label, val) in zip([c1, c2, c3, c4], metrics):
        col.markdown(
            f"<div class='metric-card'><div style='font-size:2rem'>{icon}</div>"
            f"<div style='font-size:1.7rem;font-weight:700;color:#58a6ff'>{val}</div>"
            f"<div style='color:#8b949e;font-size:0.85rem'>{label}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 2. Global Keywords & Word Cloud
    col_wc, col_bar = st.columns([1, 1])

    with col_wc:
        st.markdown("### ☁️ Word Cloud — Top Keywords")
        kw_dict = {w: s for w, s in global_keywords}
        wc = WordCloud(
            width=700, height=400,
            background_color="#0d1117",
            colormap="Blues",
            max_words=60,
            prefer_horizontal=0.8,
        ).generate_from_frequencies(kw_dict)
        fig_wc, ax_wc = plt.subplots(figsize=(7, 4))
        fig_wc.patch.set_facecolor("#0d1117")
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)

    with col_bar:
        st.markdown("### 🏆 Top 20 Keywords by TF-IDF Score")
        top20 = global_keywords[:20]
        kw_df = pd.DataFrame(top20, columns=["keyword", "score"])
        fig_bar = px.bar(
            kw_df, x="score", y="keyword", orientation="h",
            color="score", color_continuous_scale="Blues", template="plotly_dark",
        )
        fig_bar.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False, coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=10, t=10, b=10), height=400,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # 3. Topic Distribution
    st.markdown("## 🗂️ Discovered Topics")
    topic_counts = df["dominant_topic"].value_counts().reset_index()
    topic_counts.columns = ["topic_id", "count"]
    topic_labels = {t["topic_id"]: t["label"] for t in topics}
    topic_counts["label"] = topic_counts["topic_id"].map(topic_labels)
    topic_counts["pct"] = (topic_counts["count"] / len(df) * 100).round(1)

    fig_pie = px.pie(
        topic_counts, values="count", names="label",
        color_discrete_sequence=px.colors.sequential.Blues_r,
        template="plotly_dark",
    )
    fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          legend=dict(bgcolor="rgba(0,0,0,0)"), height=350)

    pie_col, tbl_col = st.columns([1, 1.3])
    with pie_col:
        st.plotly_chart(fig_pie, use_container_width=True)
    with tbl_col:
        display_df = topic_counts[["label", "count", "pct"]].rename(
            columns={"label": "Topic", "count": "Papers", "pct": "% Share"})
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=300)

    st.markdown("---")

    # 4. Topic Details + Summaries
    st.markdown("## 🧩 Topic Details & Extractive Summaries")

    for topic in topics:
        tid = topic["topic_id"]
        label = topic["label"]
        top_words = topic["top_words"]
        summary = topic_summaries.get(tid, "No summary available.")
        count = int((df["dominant_topic"] == tid).sum())

        with st.expander(f"**{label}** — {count} papers", expanded=(tid == 0)):
            kw_html = "".join(
                f"<span class='keyword-chip'>{w} <small>({s:.2f})</small></span>"
                for w, s in top_words[:12]
            )
            st.markdown(f"<div style='margin-bottom:0.8rem'>{kw_html}</div>", unsafe_allow_html=True)
            st.markdown("**Extractive Summary:**")
            st.markdown(f"<div class='summary-box'>{summary}</div>", unsafe_allow_html=True)

            sample_papers = df[df["dominant_topic"] == tid][["title", "categories", "update_date"]].head(5)
            if not sample_papers.empty:
                st.markdown("**Sample Papers:**")
                st.dataframe(sample_papers.rename(columns={"update_date": "Date"}),
                             use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── 5. Raw Data Explorer
    with st.expander("📋 Explore Raw Dataset"):
        st.dataframe(
            df[["id", "title", "categories", "dominant_topic", "update_date"]].rename(
                columns={"dominant_topic": "Topic ID"}
            ),
            use_container_width=True, hide_index=True, height=350,
        )
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV", data=csv,
            file_name=f"arxiv_topics_{keyword.replace(' ', '_')}.csv",
            mime="text/csv",
        )

else:
    # Welcome screen
    st.markdown(
        """
        <div style='text-align:center; padding: 4rem 2rem;'>
            <div style='font-size:4rem; margin-bottom:1rem'>🔬</div>
            <h2 style='color:#58a6ff; margin-bottom:0.5rem'>Ready to analyze ArXiv papers</h2>
            <p style='color:#8b949e; font-size:1.05rem; max-width:600px; margin:auto'>
                Enter a research topic in the sidebar, configure your analysis parameters,
                and click <strong>Run Analysis</strong> to discover topics, keywords,
                and extractive summaries — all using traditional NLP techniques.
            </p>
            <br>
            <div style='display:flex; justify-content:center; gap:2rem; flex-wrap:wrap; margin-top:1rem'>
                <div style='background:rgba(33,38,45,0.8);border:1px solid #30363d;border-radius:10px;padding:1rem 1.5rem;width:180px'>
                    <div style='font-size:1.8rem'>📄</div>
                    <div style='color:#58a6ff;font-weight:600'>TF-IDF</div>
                    <div style='color:#8b949e;font-size:0.82rem'>Feature Extraction</div>
                </div>
                <div style='background:rgba(33,38,45,0.8);border:1px solid #30363d;border-radius:10px;padding:1rem 1.5rem;width:180px'>
                    <div style='font-size:1.8rem'>🗂️</div>
                    <div style='color:#58a6ff;font-weight:600'>LDA / K-Means</div>
                    <div style='color:#8b949e;font-size:0.82rem'>Topic Modeling</div>
                </div>
                <div style='background:rgba(33,38,45,0.8);border:1px solid #30363d;border-radius:10px;padding:1rem 1.5rem;width:180px'>
                    <div style='font-size:1.8rem'>📝</div>
                    <div style='color:#58a6ff;font-weight:600'>Extractive</div>
                    <div style='color:#8b949e;font-size:0.82rem'>Summarization</div>
                </div>
                <div style='background:rgba(33,38,45,0.8);border:1px solid #30363d;border-radius:10px;padding:1rem 1.5rem;width:180px'>
                    <div style='font-size:1.8rem'>☁️</div>
                    <div style='color:#58a6ff;font-weight:600'>Word Cloud</div>
                    <div style='color:#8b949e;font-size:0.82rem'>Visualization</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
