import pickle
from utils.text_preprocessor import preprocess_text

model = pickle.load(open('models/model.pkl', 'rb'))
vec = pickle.load(open('models/vectorizer.pkl', 'rb'))

texts = [
    """The Election Commission of India (ECI) on Saturday announced the dates for the Lok Sabha elections 2024. Chief Election Commissioner Rajiv Kumar said the polls will be held in seven phases starting April 19. The counting of votes will take place on June 4. The Model Code of Conduct comes into effect immediately with the announcement of the election schedule.""",
    """The Indian Space Research Organisation (ISRO) successfully launched its latest communication satellite, GSAT-24, from Kourou in French Guiana. The satellite, built by ISRO for NewSpace India Limited (NSIL), will provide DTH services to Tata Play.""",
    """The Ministry of Finance has released the latest GDP growth figures, indicating a strong economic recovery. Finance Minister Nirmala Sitharaman stated that the country's economy grew by 8.4% in the third quarter of the current fiscal year, exceeding expectations."""
]

# Get class indices from model.classes_
classes = list(model.classes_)
fake_idx = classes.index("FAKE")
real_idx = classes.index("REAL")

# Access intercept_ from the underlying LinearSVC estimator
underlying = model.calibrated_classifiers_[0].estimator
print(f"Bias: {underlying.intercept_[0]:.4f}")

for i, text in enumerate(texts):
    cleaned = preprocess_text(text)
    v = vec.transform([cleaned])
    pred = model.predict(v)[0]
    probs = model.predict_proba(v)[0]
    
    print(f"\n--- Text {i+1} ---")
    print(f"Prediction: {pred}")
    print(f"Probabilities: FAKE={probs[fake_idx]:.4f}, REAL={probs[real_idx]:.4f}")

