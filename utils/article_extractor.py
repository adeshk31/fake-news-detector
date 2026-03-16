import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def get_requests_session():
    """Create a requests session with retry logic and longer timeouts."""
    session = requests.Session()
    
    # Retry strategy: 3 retries, backoff factor of 1 second (1, 2, 4 seconds)
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Update common headers to look like a standard browser
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    })
    
    return session


def extract_article(url):
    """
    Extract article text from a URL.
    Strategy:
      1. Try newspaper3k (best for news sites)
      2. Fall back to BeautifulSoup <p> tag extraction
    Returns the extracted text or raises an error.
    """
    text = ""
    session = get_requests_session()

    # ── Strategy 1: newspaper3k ────────────────────────────
    try:
        import newspaper
        from newspaper import Article, Config

        # Configure newspaper with custom user agent and timeout
        config = Config()
        config.browser_user_agent = session.headers["User-Agent"]
        config.request_timeout = 20  # Increased timeout to 20 seconds
        
        article = Article(url, config=config)
        article.download()
        article.parse()
        text = article.text.strip()

        if len(text) > 200:
            logger.info(f"newspaper3k extracted {len(text)} chars from {url}")
            return text
        else:
            logger.warning(f"newspaper3k got only {len(text)} chars, trying fallback...")
    except ImportError:
        logger.warning("newspaper3k not installed, using BeautifulSoup fallback")
    except Exception as e:
        logger.warning(f"newspaper3k failed: {e}, trying fallback...")

    # ── Strategy 2: BeautifulSoup fallback ─────────────────
    try:
        # Increased timeout to 25 seconds for slow networks
        response = session.get(url, timeout=25)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style tags
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Try <article> tag first (common in news sites)
        article_tag = soup.find("article")
        if article_tag:
            paragraphs = article_tag.find_all("p")
        else:
            paragraphs = soup.find_all("p")

        text = " ".join(p.get_text().strip() for p in paragraphs if p.get_text().strip())

        if len(text) > 100:
            logger.info(f"BeautifulSoup extracted {len(text)} chars from {url}")
            return text
        else:
            logger.warning(f"BeautifulSoup got only {len(text)} chars")
            return text

    except requests.exceptions.Timeout as e:
        logger.error(f"URL fetch timed out {url}: {e}")
        raise ValueError("Connection timed out. The website took too long to respond. You can try pasting the text manually.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch URL {url}: {e}")
        raise ValueError(f"Could not fetch the article. Error: {e}")
    except Exception as e:
        logger.error(f"Failed to parse article from {url}: {e}")
        raise ValueError(f"Could not extract article text. Error: {e}")