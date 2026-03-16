import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))

# Add domain-specific stopwords to remove dataset-specific bias but keep journalistic text
domain_stopwords = {
    'reuters', 'pic', 'twitter', 'com', 'video', 'watch', 'image', 'via', 
    'getty', 'images', 'subscribe', 'read', 'more', 'newsletter', 
    'mr', 'mrs', 'ms', 'dr', 'prof'
}
stop_words.update(domain_stopwords)

# Gracefully handle missing wordnet (network issues during download)
try:
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    # Test that wordnet data is actually available
    lemmatizer.lemmatize("testing")
    USE_LEMMATIZER = True
except Exception:
    USE_LEMMATIZER = False
    lemmatizer = None


def preprocess_text(text):
    """
    Preprocess text for fake news detection.
    Steps: lowercase -> remove URLs -> remove special chars ->
           remove stopwords -> lemmatize (if available)
    """
    # Handle NaN / non-string input
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove special characters and digits (keep only letters and spaces)
    text = re.sub(r"[^a-z\s]", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize, remove stopwords, and optionally lemmatize
    words = text.split()
    if USE_LEMMATIZER:
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    else:
        words = [word for word in words if word not in stop_words and len(word) > 2]

    return " ".join(words)