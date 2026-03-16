import pickle
from utils.article_extractor import extract_article
from utils.text_preprocessor import preprocess_text

model = pickle.load(open('models/model.pkl', 'rb'))
vec = pickle.load(open('models/vectorizer.pkl', 'rb'))

# Get class indices from model.classes_
classes = list(model.classes_)
fake_idx = classes.index("FAKE")
real_idx = classes.index("REAL")

# Access coef_ from the underlying LinearSVC estimator
underlying = model.calibrated_classifiers_[0].estimator

urls = [
    "https://www.bbc.com/news/world-middle-east-68534703",
    "https://edition.cnn.com/2024/03/13/politics/biden-election-2024/index.html",
    "https://www.nytimes.com/2024/03/13/us/politics/biden-trump-2024.html",
    "https://www.reuters.com/world/us/biden-trump-clinch-nominations-kicking-off-grueling-rematch-2024-03-13/",
    "https://timesofindia.indiatimes.com/india/lok-sabha-elections-2024-dates-announcement-live-updates-eci-press-conference/liveblog/108518974.cms",
    "https://www.ndtv.com/india-news/pm-narendra-modi-in-kanyakumari-live-updates-5246733"
]

for url in urls:
    print(f"\n--- URL: {url} ---")
    try:
        text = extract_article(url)
        print(f"Extracted {len(text)} chars.")
        if len(text) < 200:
            print("Text too short!")
            continue
            
        cleaned = preprocess_text(text)
        v = vec.transform([cleaned])
        pred = model.predict(v)[0]
        probs = model.predict_proba(v)[0]
        
        print(f"Prediction: {pred}")
        print(f"Probabilities: FAKE={probs[fake_idx]:.4f}, REAL={probs[real_idx]:.4f}")
        
        feature_names = vec.get_feature_names_out()
        coefs = underlying.coef_[0]
        indices = v.nonzero()[1]
        word_scores = [(feature_names[i], coefs[i] * v[0, i], coefs[i], v[0, i]) for i in indices]
        
        real_contrib = sum(w[1] for w in word_scores if w[1] > 0)
        fake_contrib = sum(w[1] for w in word_scores if w[1] < 0)
        print(f"Total REAL vocab score: {real_contrib:.4f}")
        print(f"Total FAKE vocab score: {fake_contrib:.4f}")
        
    except Exception as e:
        print(f"Failed: {e}")
