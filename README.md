# Fake News Detection System

## 📌 Project Overview
The Fake News Detection System is a Machine Learning web application built with Python and Streamlit. It allows users to input a news article's text or URL to predict whether the news is **REAL** or **FAKE**. 

## 🏗️ System Architecture
The system follows a standard NLP pipeline:
1. **Input Interface:** User provides a news URL or raw text via the Streamlit UI.
2. **Article Extraction:** If a URL is provided, the system scrapes the article's main body text.
3. **Text Preprocessing:** The raw text is cleaned (lowercased, URLs/emails removed, punctuation stripped, stopwords removed, and lemmatized).
4. **Feature Extraction:** The cleaned text is converted into numerical vectors using TF-IDF.
5. **Classification:** A trained Logistic Regression model evaluates the vector and outputs a probability score for REAL vs. FAKE.
6. **Output Interface:** The Streamlit app displays the prediction, confidence score, and the most important keywords that influenced the decision.

## 🛠️ Technologies Used
* **Programming Language:** Python 3.8+
* **Web Framework:** Streamlit
* **Machine Learning:** Scikit-learn, Pandas, NumPy
* **Natural Language Processing (NLP):** NLTK (Natural Language Toolkit)
* **Web Scraping:** Newspaper3k, BeautifulSoup4, Requests

## 📦 Modules of the System

### 1. Dataset Preparation (`prepare_dataset.py`)
* Loads raw `True.csv` and `Fake.csv` datasets.
* **Crucial Fix:** Strips source metadata (e.g., `WASHINGTON (Reuters) -`) to prevent data leakage where the model learns to identify the publisher rather than the news content.
* Combines, shuffles, and saves the cleaned dataset for training.

### 2. Text Preprocessor (`utils/text_preprocessor.py`)
* Converts text to lowercase.
* Removes URLs, email addresses, digits, and special characters.
* Removes common English stopwords using NLTK.
* Applies WordNet Lemmatization to reduce words to their base form.

### 3. Article Extractor (`utils/article_extractor.py`)
* Uses `newspaper3k` as the primary method to scrape and parse articles from URLs.
* Includes a fallback mechanism using `BeautifulSoup4` and the `requests` library in case `newspaper3k` fails.
* Cleans HTML tags, HTTP headers, and navigational elements to isolate the main article text.

### 4. Model Training (`train_model.py`)
* Vectorizes the preprocessed text using `TfidfVectorizer` (with unigrams, bigrams, and sublinear TF).
* Splits the data into training (80%) and testing (20%) sets.
* Trains a `LogisticRegression` model.
* Evaluates the model (Accuracy, Classification Report, Confusion Matrix).
* Saves the trained model (`model.pkl`) and vectorizer (`vectorizer.pkl`) for deployment.

### 5. Web Application (`app.py`)
* Interactive Streamlit interface.
* Accepts URLs or direct text pasting.
* Dynamically extracts, cleans, vectorizes, and predicts in real-time.
* Displays a detailed visual breakdown of prediction probabilities and influential keywords.

## 🧠 Algorithms Used

### 1. TF-IDF (Term Frequency - Inverse Document Frequency)
* **Purpose:** Feature extraction.
* Converts raw text into a matrix of TF-IDF features. It measures how important a word is to a document in a collection. We use bigrams `ngram_range=(1,2)` to capture context (e.g., "climate change" instead of just "climate" and "change").

### 2. Logistic Regression
* **Purpose:** Binary Classification.
* A statistical model used to predict the probability of a binary outcome (0: FAKE, 1: REAL). It is highly efficient and interpretable for sparse textual data produced by TF-IDF.

## 🚀 Working of the Project (How to Run)

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download NLTK Data (First-time setup):**
   ```python
   python -c "import nltk; nltk.download('wordnet'); nltk.download('stopwords')"
   ```

3. **Prepare the Dataset:**
   ```bash
   python prepare_dataset.py
   ```

4. **Train the Model:**
   ```bash
   python train_model.py
   ```

5. **Run the Streamlit Web App:**
   ```bash
   streamlit run app.py
   ```
