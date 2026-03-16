import pandas as pd
import pickle
import logging

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils.text_preprocessor import preprocess_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Load data ──────────────────────────────────────────────
logger.info("Loading dataset...")
data = pd.read_csv("data/news_dataset.csv")
logger.info(f"Dataset loaded: {len(data)} samples")
logger.info(f"  FAKE: {(data['label'] == 'FAKE').sum()}")
logger.info(f"  REAL: {(data['label'] == 'REAL').sum()}")


# ── Preprocess ─────────────────────────────────────────────
logger.info("Applying text preprocessing (this may take a few minutes)...")
data["text"] = data["text"].apply(preprocess_text)

# Drop rows that became empty after preprocessing
data = data[data["text"].str.strip().astype(bool)]
logger.info(f"After preprocessing: {len(data)} samples remain")


# ── Vectorize ──────────────────────────────────────────────
X = data["text"]
y = data["label"]

# TF-IDF configuration:
# - stop_words="english": sklearn's built-in English stopwords (in addition to
#   the NLTK stopwords already removed during preprocessing)
# - max_df=0.7: ignore terms that appear in more than 70% of docs (too common)
# - min_df=5: ignore terms that appear in fewer than 5 docs (too rare / noisy)
# - ngram_range=(1,2): unigrams + bigrams for richer feature space
# - sublinear_tf=True: log-scale term frequency (reduces impact of very common terms)
# - max_features=100000: cap vocabulary to keep memory manageable
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    min_df=5,
    ngram_range=(1, 2),
    sublinear_tf=True,
    max_features=100000,
)

logger.info("Fitting TF-IDF vectorizer...")
X_vec = vectorizer.fit_transform(X)
logger.info(f"Vocabulary size: {len(vectorizer.vocabulary_)}")


# ── Train / Test Split ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)
logger.info(f"Train set: {X_train.shape[0]}  |  Test set: {X_test.shape[0]}")


# ── Train Model ────────────────────────────────────────────
# LinearSVC: fastest, best-performing algorithm for sparse TF-IDF text data.
# Wrapped with CalibratedClassifierCV so we get predict_proba() support.
logger.info("Training LinearSVC model...")
svc = LinearSVC(max_iter=2000, C=1.0)
model = CalibratedClassifierCV(svc, cv=5)
model.fit(X_train, y_train)

logger.info(f"Model classes: {model.classes_}")


# ── Evaluate ───────────────────────────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

logger.info(f"\n{'='*50}")
logger.info(f"TEST ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
logger.info(f"{'='*50}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=["FAKE", "REAL"])
print(f"  Predicted:   FAKE    REAL")
print(f"  Actual FAKE: {cm[0][0]:>5}   {cm[0][1]:>5}")
print(f"  Actual REAL: {cm[1][0]:>5}   {cm[1][1]:>5}")


# ── Save Model & Vectorizer ───────────────────────────────
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

logger.info("\nModel saved     → models/model.pkl")
logger.info("Vectorizer saved → models/vectorizer.pkl")
logger.info("Training complete!")