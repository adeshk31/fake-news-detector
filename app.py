import streamlit as st
import pickle
import logging
import time
import nltk
from urllib.parse import urlparse

# Download NLTK data (required on Streamlit Cloud)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from utils.article_extractor import extract_article
from utils.text_preprocessor import preprocess_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Page Configuration ─────────────────────────────────────
st.set_page_config(
    page_title="VerifyNow — AI Fake News Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get help": None,
        "Report a Bug": None,
        "About": "VerifyNow — AI-Powered Fake News Detection System. "
                 "Uses a Linear SVM model trained on 44,000+ articles "
                 "combined with domain credibility analysis.",
    },
)


# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ===== IMPORTS ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ===== GLOBAL RESET ===== */
    *, *::before, *::after { box-sizing: border-box; }

    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(145deg, #0a0a1a 0%, #0d1b2a 30%, #1b2838 60%, #0a0a1a 100%);
        color: #e0e6ed;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ===== HERO SECTION ===== */
    .hero-container {
        text-align: center;
        padding: 2rem 1rem 1.5rem;
        position: relative;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 179, 237, 0.08);
        border: 1px solid rgba(99, 179, 237, 0.2);
        border-radius: 100px;
        padding: 6px 16px;
        font-size: 0.75rem;
        font-weight: 500;
        color: #63b3ed;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 0.75rem;
        background: linear-gradient(135deg, #ffffff 0%, #a0c4ff 50%, #63b3ed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #8899a6;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
        font-weight: 400;
    }

    /* ===== GLASS CARD ===== */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 1.75rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(99, 179, 237, 0.15);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }

    .card-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1rem;
    }
    .card-header-icon {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        flex-shrink: 0;
    }
    .card-header-icon.blue { background: rgba(99, 179, 237, 0.12); }
    .card-header-icon.purple { background: rgba(159, 122, 234, 0.12); }
    .card-header-icon.green { background: rgba(72, 187, 120, 0.12); }

    .card-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #c9d1d9;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* ===== RESULT VERDICT CARDS ===== */
    .verdict-card {
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        margin-bottom: 1rem;
    }
    .verdict-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
    }
    .verdict-real {
        background: linear-gradient(135deg, rgba(72, 187, 120, 0.08) 0%, rgba(56, 161, 105, 0.04) 100%);
        border: 1px solid rgba(72, 187, 120, 0.2);
    }
    .verdict-real::before {
        background: linear-gradient(90deg, #48bb78, #38a169);
    }
    .verdict-fake {
        background: linear-gradient(135deg, rgba(245, 101, 101, 0.08) 0%, rgba(229, 62, 62, 0.04) 100%);
        border: 1px solid rgba(245, 101, 101, 0.2);
    }
    .verdict-fake::before {
        background: linear-gradient(90deg, #f56565, #e53e3e);
    }
    .verdict-uncertain {
        background: linear-gradient(135deg, rgba(236, 201, 75, 0.08) 0%, rgba(214, 158, 46, 0.04) 100%);
        border: 1px solid rgba(236, 201, 75, 0.2);
    }
    .verdict-uncertain::before {
        background: linear-gradient(90deg, #ecc94b, #d69e2e);
    }

    .verdict-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .verdict-label {
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.25rem;
    }
    .verdict-real .verdict-label { color: #68d391; }
    .verdict-fake .verdict-label { color: #fc8181; }
    .verdict-uncertain .verdict-label { color: #ecc94b; }

    .verdict-title {
        font-size: 1.6rem;
        font-weight: 800;
    }
    .verdict-real .verdict-title { color: #48bb78; }
    .verdict-fake .verdict-title { color: #f56565; }
    .verdict-uncertain .verdict-title { color: #ecc94b; }

    .verdict-note {
        font-size: 0.85rem;
        color: #8899a6;
        margin-top: 0.5rem;
        line-height: 1.5;
    }

    /* ===== METRICS ROW ===== */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        transition: all 0.25s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 179, 237, 0.15);
    }
    .metric-label {
        font-size: 0.7rem;
        font-weight: 500;
        color: #6b7a8d;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
    }
    .metric-value.green { color: #48bb78; }
    .metric-value.red { color: #f56565; }
    .metric-value.blue { color: #63b3ed; }
    .metric-value.yellow { color: #ecc94b; }
    .metric-value.white { color: #e2e8f0; }

    /* ===== PROBABILITY BAR ===== */
    .prob-container {
        margin-bottom: 1rem;
    }
    .prob-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .prob-label {
        font-size: 0.8rem;
        font-weight: 500;
        color: #a0aec0;
    }
    .prob-value {
        font-size: 0.8rem;
        font-weight: 600;
    }
    .prob-bar-track {
        width: 100%;
        height: 8px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 100px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 100px;
        transition: width 1s ease-out;
    }
    .prob-bar-fill.green {
        background: linear-gradient(90deg, #48bb78, #38a169);
    }
    .prob-bar-fill.red {
        background: linear-gradient(90deg, #f56565, #e53e3e);
    }

    /* ===== INFLUENTIAL WORDS ===== */
    .word-chip {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: rgba(99, 179, 237, 0.08);
        border: 1px solid rgba(99, 179, 237, 0.15);
        border-radius: 8px;
        padding: 6px 12px;
        margin: 3px;
        font-size: 0.8rem;
        font-weight: 500;
        color: #a0c4ff;
        transition: all 0.2s ease;
    }
    .word-chip:hover {
        background: rgba(99, 179, 237, 0.14);
        transform: translateY(-1px);
    }
    .word-score {
        font-size: 0.65rem;
        color: #63b3ed;
        opacity: 0.7;
    }

    /* ===== SOURCE BADGE ===== */
    .source-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 8px 14px;
        border-radius: 10px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .source-high {
        background: rgba(72, 187, 120, 0.1);
        border: 1px solid rgba(72, 187, 120, 0.2);
        color: #68d391;
    }
    .source-medium {
        background: rgba(236, 201, 75, 0.1);
        border: 1px solid rgba(236, 201, 75, 0.2);
        color: #ecc94b;
    }
    .source-low {
        background: rgba(245, 101, 101, 0.1);
        border: 1px solid rgba(245, 101, 101, 0.2);
        color: #fc8181;
    }
    .source-unknown {
        background: rgba(160, 174, 192, 0.08);
        border: 1px solid rgba(160, 174, 192, 0.15);
        color: #a0aec0;
    }

    /* ===== ARTICLE PREVIEW ===== */
    .preview-text {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1.25rem;
        font-size: 0.88rem;
        line-height: 1.7;
        color: #a0aec0;
        max-height: 300px;
        overflow-y: auto;
    }

    /* ===== HOW IT WORKS ===== */
    .steps-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-top: 1rem;
    }
    .step-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .step-card:hover {
        border-color: rgba(99, 179, 237, 0.15);
        transform: translateY(-3px);
    }
    .step-num {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: linear-gradient(135deg, rgba(99, 179, 237, 0.15), rgba(159, 122, 234, 0.15));
        border: 1px solid rgba(99, 179, 237, 0.2);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 700;
        color: #63b3ed;
        margin-bottom: 0.75rem;
    }
    .step-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #c9d1d9;
        margin-bottom: 0.4rem;
    }
    .step-desc {
        font-size: 0.75rem;
        color: #6b7a8d;
        line-height: 1.5;
    }

    /* ===== FOOTER ===== */
    .app-footer {
        text-align: center;
        padding: 2rem 0 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.04);
        margin-top: 2rem;
    }
    .footer-text {
        font-size: 0.75rem;
        color: #4a5568;
        line-height: 1.8;
    }

    /* ===== STREAMLIT OVERRIDES ===== */
    .stTextInput > div > div {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.2s ease !important;
    }
    .stTextInput > div > div:focus-within {
        border-color: rgba(99, 179, 237, 0.4) !important;
        box-shadow: 0 0 0 3px rgba(99, 179, 237, 0.08) !important;
    }

    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease !important;
    }
    .stTextArea > div > div > textarea:focus {
        border-color: rgba(99, 179, 237, 0.4) !important;
        box-shadow: 0 0 0 3px rgba(99, 179, 237, 0.08) !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #3182ce 0%, #2b6cb0 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 2rem !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.02em !important;
        width: 100% !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 15px rgba(49, 130, 206, 0.25) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2b6cb0 0%, #2c5282 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(49, 130, 206, 0.35) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        color: #6b7a8d !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        padding: 8px 16px !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99, 179, 237, 0.1) !important;
        color: #63b3ed !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.02) !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        color: #8899a6 !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }

    /* Divider */
    hr {
        border-color: rgba(255, 255, 255, 0.04) !important;
    }

    /* Alert overrides */
    .stAlert > div {
        border-radius: 12px !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #63b3ed !important;
    }

    /* Responsive */
    @media (max-width: 768px) {
        .hero-title { font-size: 2rem; }
        .metrics-grid { grid-template-columns: 1fr; }
        .steps-grid { grid-template-columns: repeat(2, 1fr); }
    }
</style>
""", unsafe_allow_html=True)


# ── Load Model ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("models/model.pkl", "rb"))
        vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
        logger.info(f"Model loaded. Classes: {model.classes_}")
        return model, vectorizer
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please run `python train_model.py` first.")
        st.stop()


model, vectorizer = load_model()


# ── Helper Functions ───────────────────────────────────────
def get_top_words(vector, vectorizer, n=10):
    feature_names = vectorizer.get_feature_names_out()
    scores = vector.toarray()[0]
    top_indices = scores.argsort()[-n:][::-1]
    return [
        (feature_names[i], scores[i])
        for i in top_indices if scores[i] > 0
    ]


# Domain Credibility Database
SOURCE_CREDIBILITY = {
    # ── Highly Credible ─────────────────────────────────────
    # International wire services & newspapers of record
    "bbc.com": ("Highly Credible", "high"),
    "bbc.co.uk": ("Highly Credible", "high"),
    "reuters.com": ("Highly Credible", "high"),
    "apnews.com": ("Highly Credible", "high"),
    "nytimes.com": ("Highly Credible", "high"),
    "theguardian.com": ("Highly Credible", "high"),
    "washingtonpost.com": ("Highly Credible", "high"),
    "aljazeera.com": ("Highly Credible", "high"),
    "dw.com": ("Highly Credible", "high"),
    "france24.com": ("Highly Credible", "high"),
    "abc.net.au": ("Highly Credible", "high"),
    "economist.com": ("Highly Credible", "high"),
    "bloomberg.com": ("Highly Credible", "high"),
    # Indian newspapers of record
    "thehindu.com": ("Highly Credible", "high"),
    "indianexpress.com": ("Highly Credible", "high"),
    "livemint.com": ("Highly Credible", "high"),
    "deccanherald.com": ("Highly Credible", "high"),
    "scroll.in": ("Highly Credible", "high"),
    "theprint.in": ("Highly Credible", "high"),
    # ── Generally Reliable ─────────────────────────────────
    # Major TV networks & large-circulation dailies
    "ndtv.com": ("Generally Reliable", "medium"),
    "timesofindia.com": ("Generally Reliable", "medium"),
    "indiatimes.com": ("Generally Reliable", "medium"),
    "hindustantimes.com": ("Generally Reliable", "medium"),
    "cnn.com": ("Generally Reliable", "medium"),
    "cbsnews.com": ("Generally Reliable", "medium"),
    "nbcnews.com": ("Generally Reliable", "medium"),
    "usatoday.com": ("Generally Reliable", "medium"),
    "forbes.com": ("Generally Reliable", "medium"),
    "news18.com": ("Generally Reliable", "medium"),
    "firstpost.com": ("Generally Reliable", "medium"),
    "zeenews.india.com": ("Generally Reliable", "medium"),
    "moneycontrol.com": ("Generally Reliable", "medium"),
    "business-standard.com": ("Generally Reliable", "medium"),
    "theeconomictimes.com": ("Generally Reliable", "medium"),
    "tribuneindia.com": ("Generally Reliable", "medium"),
    "thestatesman.com": ("Generally Reliable", "medium"),
    "telegraphindia.com": ("Generally Reliable", "medium"),
    "aajtak.in": ("Generally Reliable", "medium"),
    "abplive.com": ("Generally Reliable", "medium"),
    # Sports-specific credible sources
    "espncricinfo.com": ("Generally Reliable", "medium"),
    "espn.com": ("Generally Reliable", "medium"),
    "cricbuzz.com": ("Generally Reliable", "medium"),
    "icc-cricket.com": ("Highly Credible", "high"),
    "bcci.tv": ("Highly Credible", "high"),
    "skysports.com": ("Generally Reliable", "medium"),
    "goal.com": ("Generally Reliable", "medium"),
    "sportstar.thehindu.com": ("Generally Reliable", "medium"),
    # ── Low Credibility ────────────────────────────────────
    "opindia.com": ("Low Credibility", "low"),
    "breitbart.com": ("Low Credibility", "low"),
}


# ── Hero Section ───────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">🛡️ AI-Powered Verification Engine</div>
    <div class="hero-title">VerifyNow</div>
    <p class="hero-subtitle">
        Analyze any news article in seconds. Our hybrid AI engine combines 
        machine learning with source credibility analysis to detect misinformation.
    </p>
</div>
""", unsafe_allow_html=True)


# ── Input Section ──────────────────────────────────────────
st.markdown("""
<div class="glass-card" style="max-width: 800px; margin: 0 auto 1.5rem;">
    <div class="card-header">
        <div class="card-header-icon blue">🔗</div>
        <span class="card-title">Analyze an Article</span>
    </div>
</div>
""", unsafe_allow_html=True)

col_input_l, col_input_m, col_input_r = st.columns([1, 4, 1])
with col_input_m:
    input_tab1, input_tab2 = st.tabs(["🔗  Paste URL", "📝  Paste Text"])

    with input_tab1:
        url = st.text_input(
            "Article URL",
            placeholder="https://www.bbc.com/news/...",
            label_visibility="collapsed",
        )

    with input_tab2:
        news_text = st.text_area(
            "Article Text",
            placeholder="Paste the full article text here...",
            height=180,
            label_visibility="collapsed",
        )

    analyze_btn = st.button("🔍  Analyze Article", use_container_width=True)


# ── Analysis Pipeline ─────────────────────────────────────
if analyze_btn:
    text = ""

    # Determine input source
    if news_text and news_text.strip():
        text = news_text.strip()
    elif url and url.strip():
        try:
            with st.spinner("Extracting article content..."):
                text = extract_article(url.strip())
        except ValueError as e:
            st.error(f"**Extraction Failed:** {e}")
            st.stop()
    else:
        st.warning("Please provide a URL or paste article text to analyze.")
        st.stop()

    if len(text) < 150:
        st.warning(
            f"Article text is too short ({len(text)} characters). "
            "Please provide the full article body for accurate analysis."
        )
        st.stop()

    # ── Run ML Pipeline ───────────────────────────────────
    with st.spinner("Analyzing text patterns..."):
        cleaned_text = preprocess_text(text)
        vector = vectorizer.transform([cleaned_text])
        prediction = model.predict(vector)[0]
        probabilities = model.predict_proba(vector)[0]

        classes = list(model.classes_)
        fake_idx = classes.index("FAKE")
        real_idx = classes.index("REAL")

        fake_prob = probabilities[fake_idx]
        real_prob = probabilities[real_idx]
        confidence = max(fake_prob, real_prob) * 100

        # Small delay for UX feel
        time.sleep(0.3)

    # ── Source Credibility ────────────────────────────────
    source_domain = "Manual Input"
    source_label = "Unknown"
    source_tier = "unknown"
    is_trusted = False
    is_low_cred = False

    if url and url.strip():
        parsed = urlparse(url.strip())
        domain = parsed.netloc.replace("www.", "")
        source_domain = domain

        matched = next(
            (v for k, v in SOURCE_CREDIBILITY.items() if domain.endswith(k)),
            None,
        )
        if matched:
            source_label, source_tier = matched
            is_trusted = source_tier in ("high", "medium")
            is_low_cred = source_tier == "low"

    # ── Final Verdict ─────────────────────────────────────
    if is_trusted:
        final_verdict = "Likely REAL"
        final_note = (
            f"The source ({source_domain}) is recognized as a "
            f"credible news organization."
        )
        verdict_class = "real"
        verdict_icon = "✅"
    elif is_low_cred:
        final_verdict = "Likely FAKE"
        final_note = (
            f"The source ({source_domain}) has a history of "
            f"publishing unverified or biased content."
        )
        verdict_class = "fake"
        verdict_icon = "🚨"
    else:
        # For unknown sources, use confidence threshold to avoid false verdicts
        if confidence < 70:
            final_verdict = "Uncertain"
            final_note = (
                "ML confidence is low and the source is not in our credibility "
                "database. Please verify this article through multiple trusted sources."
            )
            verdict_class = "uncertain"
            verdict_icon = "⚠️"
        else:
            final_verdict = f"Likely {prediction}"
            final_note = (
                "Assessment based on AI content analysis. "
                "Source credibility is unknown."
            )
            verdict_class = "real" if prediction == "REAL" else "fake"
            verdict_icon = "✅" if prediction == "REAL" else "🚨"

    # ── RESULTS UI ────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)

    # Verdict Card
    st.markdown(f"""
    <div class="verdict-card verdict-{verdict_class}">
        <div class="verdict-icon">{verdict_icon}</div>
        <div class="verdict-label">Verification Result</div>
        <div class="verdict-title">{final_verdict}</div>
        <div class="verdict-note">{final_note}</div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics Row
    pred_color = "green" if prediction == "REAL" else "red"
    conf_color = "green" if confidence >= 75 else ("yellow" if confidence >= 50 else "red")

    source_badge_class = f"source-{source_tier}"
    source_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴", "unknown": "⚪"}[source_tier]

    st.markdown(f"""
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">ML Prediction</div>
            <div class="metric-value {pred_color}">{prediction}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Confidence Score</div>
            <div class="metric-value {conf_color}">{confidence:.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Source Credibility</div>
            <div class="source-badge {source_badge_class}">
                {source_emoji} {source_label}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Probability Bars + Details
    col_left, col_right = st.columns([3, 2])

    with col_left:
        # Probability Breakdown
        st.markdown("""
        <div class="glass-card">
            <div class="card-header">
                <div class="card-header-icon green">📊</div>
                <span class="card-title">Probability Breakdown</span>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="prob-container">
                <div class="prob-header">
                    <span class="prob-label">✅ Real Probability</span>
                    <span class="prob-value" style="color: #48bb78;">{real_prob*100:.1f}%</span>
                </div>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill green" style="width: {real_prob*100:.1f}%;"></div>
                </div>
            </div>
            <div class="prob-container">
                <div class="prob-header">
                    <span class="prob-label">🚨 Fake Probability</span>
                    <span class="prob-value" style="color: #f56565;">{fake_prob*100:.1f}%</span>
                </div>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill red" style="width: {fake_prob*100:.1f}%;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Article Preview
        st.markdown("""
        <div class="glass-card">
            <div class="card-header">
                <div class="card-header-icon purple">📄</div>
                <span class="card-title">Article Preview</span>
            </div>
        """, unsafe_allow_html=True)

        preview = text[:600] + ("..." if len(text) > 600 else "")
        # Escape HTML in preview text
        preview = preview.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(f"""
            <div class="preview-text">{preview}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        # Influential Words
        st.markdown("""
        <div class="glass-card">
            <div class="card-header">
                <div class="card-header-icon blue">🔑</div>
                <span class="card-title">Key Signal Words</span>
            </div>
        """, unsafe_allow_html=True)

        words = get_top_words(vector, vectorizer)
        if words:
            chips_html = ""
            for word, score in words:
                chips_html += f'<span class="word-chip">{word} <span class="word-score">{score:.3f}</span></span>'
            st.markdown(f"""
                <div style="margin-top: 0.25rem;">{chips_html}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <p style="color: #6b7a8d; font-size: 0.85rem;">No strong signal words detected.</p>
            </div>
            """, unsafe_allow_html=True)

        # Analysis Details
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-header">
                <div class="card-header-icon purple">🛠️</div>
                <span class="card-title">Analysis Details</span>
            </div>
            <div style="font-size: 0.82rem; color: #6b7a8d; line-height: 2;">
                <div style="display: flex; justify-content: space-between;">
                    <span>Article Length</span>
                    <span style="color: #a0aec0; font-weight: 500;">{len(text):,} chars</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>Processed Length</span>
                    <span style="color: #a0aec0; font-weight: 500;">{len(cleaned_text):,} chars</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>Feature Count</span>
                    <span style="color: #a0aec0; font-weight: 500;">{int(vector.nnz):,}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>Source Domain</span>
                    <span style="color: #a0aec0; font-weight: 500;">{source_domain}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── How It Works Section ───────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="glass-card" style="max-width: 900px; margin: 0 auto;">
    <div style="text-align: center; margin-bottom: 0.5rem;">
        <span class="card-title" style="font-size: 0.9rem;">How It Works</span>
    </div>
    <div class="steps-grid">
        <div class="step-card">
            <div class="step-num">1</div>
            <div class="step-title">Input</div>
            <div class="step-desc">Paste a URL or article text for analysis</div>
        </div>
        <div class="step-card">
            <div class="step-num">2</div>
            <div class="step-title">Extract & Clean</div>
            <div class="step-desc">NLP pipeline removes noise and normalizes text</div>
        </div>
        <div class="step-card">
            <div class="step-num">3</div>
            <div class="step-title">AI Analysis</div>
            <div class="step-desc">Linear SVM model trained on 44,000+ articles</div>
        </div>
        <div class="step-card">
            <div class="step-num">4</div>
            <div class="step-title">Verdict</div>
            <div class="step-desc">Combined ML score + source credibility check</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <p class="footer-text">
        VerifyNow — AI Fake News Detection System<br>
        Built with Streamlit • Linear SVM + TF-IDF • Trained on 44,000+ articles<br>
        <span style="color: #2d3748;">This tool is for educational purposes. Always verify news through multiple credible sources.</span>
    </p>
</div>
""", unsafe_allow_html=True)