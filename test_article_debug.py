import pickle
from utils.text_preprocessor import preprocess_text

model = pickle.load(open('models/model.pkl', 'rb'))
vec = pickle.load(open('models/vectorizer.pkl', 'rb'))

text = """Breaking: Narendra Modi Secretly Resigns From Prime Minister Post Overnight

New Delhi — In a surprising development, viral social media posts claim that Prime Minister Narendra Modi has secretly resigned from his position late last night. According to unverified sources, the resignation allegedly took place during a closed-door emergency meeting in New Delhi.

Several posts circulating online claim that senior government officials were informed about the decision but were instructed to keep it confidential until an official announcement is prepared. However, no recognized news organizations have confirmed these claims.

Political analysts warn that the viral messages appear to originate from anonymous accounts and could be part of misinformation campaigns spreading rapidly on social media."""

cleaned = preprocess_text(text)
v = vec.transform([cleaned])
pred = model.predict(v)[0]
probs = model.predict_proba(v)[0]

# Get class indices from model.classes_
classes = list(model.classes_)
fake_idx = classes.index("FAKE")
real_idx = classes.index("REAL")

# Access coef_ and intercept_ from the underlying LinearSVC estimator
# CalibratedClassifierCV wraps the estimator; access via calibrated_classifiers_
underlying = model.calibrated_classifiers_[0].estimator
feature_names = vec.get_feature_names_out()
coefs = underlying.coef_[0]

print(f"Cleaned Text: {cleaned}")
print(f"Prediction: {pred}")
print(f"Probabilities: FAKE={probs[fake_idx]:.4f}, REAL={probs[real_idx]:.4f}")

indices = v.nonzero()[1]
word_scores = [(feature_names[i], coefs[i] * v[0, i], coefs[i], v[0, i]) for i in indices]

# Bias term
bias = underlying.intercept_[0]
print(f"\nIntercept (Bias): {bias:.4f}")

print("\nTop contributing words to REAL (> 0):")
for w in sorted([ws for ws in word_scores if ws[1] > 0], key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {w[0]}: contribution={w[1]:.4f} (coef={w[2]:.4f}, tfidf={w[3]:.4f})")

print("\nTop contributing words to FAKE (< 0):")
for w in sorted([ws for ws in word_scores if ws[1] < 0], key=lambda x: x[1])[:10]:
    print(f"  {w[0]}: contribution={w[1]:.4f} (coef={w[2]:.4f}, tfidf={w[3]:.4f})")
