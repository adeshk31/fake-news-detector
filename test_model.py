import pickle
from utils.text_preprocessor import preprocess_text

# Load model and vectorizer
try:
    model = pickle.load(open("models/model.pkl", "rb"))
    vec = pickle.load(open("models/vectorizer.pkl", "rb"))
    print(f"Model classes: {model.classes_}")
except FileNotFoundError:
    print("Error: Models not found. Run train_model.py first.")
    exit()

def test_article(title, content):
    print(f"\n--- Testing: {title} ---")
    cleaned = preprocess_text(content)
    v = vec.transform([cleaned])
    prediction = model.predict(v)[0]
    probs = model.predict_proba(v)[0]
    
    # Map probabilities to classes
    prob_dict = dict(zip(model.classes_, probs))
    
    print(f"Prediction : {prediction}")
    print(f"Confidence : {prob_dict[prediction]*100:.1f}%")
    print(f"Probabilities: {prob_dict}")

# Test 1: Real factual news
real_news = """
The European Central Bank maintained its benchmark interest rate at a record high on Thursday, 
signaling that it is not yet ready to start cutting rates despite easing inflation. 
ECB President Christine Lagarde stated that policy would remain restrictive for as long as 
necessary to ensure inflation returns to its 2% target. Market analysts had widely expected 
the move, focusing instead on potential hints about a June rate cut.
"""

# Test 2: Fake inflammatory news
fake_news = """
SECRET DOCUMENTS REVEALED: The moon is actually a hollow space station built by 
ancient civilizations to monitor Earth! Mainstream scientists have been silenced 
for decades about the electromagnetic signals coming from inside the moon. 
The government is already planning to tax our lunar views to fund their 
secret space army. Share this before it gets deleted by the elites!
"""

test_article("REAL NEWS", real_news)
test_article("FAKE NEWS", fake_news)
