import pickle
from utils.text_preprocessor import preprocess_text

model = pickle.load(open('models/model.pkl', 'rb'))
vec = pickle.load(open('models/vectorizer.pkl', 'rb'))

text1 = """
WASHINGTON — President Joe Biden and former President Donald J. Trump marched to their respective parties' presidential nominations on Tuesday, setting up a grueling eight-month general election rematch that will test the country's democratic institutions.

Mr. Biden faced no major opposition in the Democratic primary, easily winning the delegates he needed. Mr. Trump, too, coasted through the Republican contests, vanquishing his last remaining rival, Nikki Haley, last week.

The official start of the general election effectively ends a primary season that lacked suspense and drama, turning the focus exclusively to the coming clash between the two men. It will be the first presidential rematch since 1956, and the first time an incumbent president has faced a former president since 1892.

Both candidates enter the general election with significant vulnerabilities: Mr. Biden faces widespread concerns about his age and his handling of the economy, while Mr. Trump is navigating four criminal indictments and questions about his legal subversion of the democratic process.
"""

cleaned = preprocess_text(text1)
v = vec.transform([cleaned])
pred = model.predict(v)[0]
probs = model.predict_proba(v)[0]

# Get class indices from model.classes_
classes = list(model.classes_)
fake_idx = classes.index("FAKE")
real_idx = classes.index("REAL")

print(f"Prediction: {pred}")
print(f"Probabilities: FAKE={probs[fake_idx]:.4f}, REAL={probs[real_idx]:.4f}")

# Access coef_ and intercept_ from the underlying LinearSVC estimator
underlying = model.calibrated_classifiers_[0].estimator
feature_names = vec.get_feature_names_out()
coefs = underlying.coef_[0]
indices = v.nonzero()[1]
word_scores = [(feature_names[i], coefs[i] * v[0, i], coefs[i], v[0, i]) for i in indices]

real_contrib = sum(w[1] for w in word_scores if w[1] > 0)
fake_contrib = sum(w[1] for w in word_scores if w[1] < 0)
print(f"Bias: {underlying.intercept_[0]:.4f}")
print(f"Total REAL vocab score: {real_contrib:.4f}")
print(f"Total FAKE vocab score: {fake_contrib:.4f}")
print("\nTop REAL words:")
for w in sorted([w for w in word_scores if w[1]>0], key=lambda x:x[1], reverse=True)[:5]:
    print(w)
print("\nTop FAKE words:")
for w in sorted([w for w in word_scores if w[1]<0], key=lambda x:x[1])[:5]:
    print(w)
