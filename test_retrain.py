import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from utils.text_preprocessor import preprocess_text

# load original vectors (if I can, wait, they are generated in train_model.py)
# Actually, I'll just run train_model but change C=0.1
