from utils.article_extractor import extract_article

url = "https://www.bbc.com/news/world-60525350"

text = extract_article(url)

print(text[:1000])