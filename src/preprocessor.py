"""
Text Preprocessor for ArXiv abstracts.
Handles cleaning, tokenization, stop-word removal, and lemmatization.
"""
import re
import string
import nltk
import spacy
from typing import List

# Download required NLTK data on first run
def download_nltk_resources():
    """Download required NLTK resources (handles SSL issues on macOS)."""
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    resources = ["punkt", "punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger"]
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

download_nltk_resources()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# Try to load spaCy model; fall back gracefully if not installed
try:
    nlp = spacy.load("en_core_web_sm")
    USE_SPACY = True
except OSError:
    USE_SPACY = False

STOP_WORDS = set(stopwords.words("english"))
# Add domain-specific stop words for scientific papers
EXTRA_STOP_WORDS = {
    "paper", "propose", "show", "result", "method", "approach",
    "present", "study", "also", "use", "used", "using", "based",
    "new", "one", "two", "may", "well", "among", "within",
    "et", "al", "fig", "eq", "arxiv", "preprint"
}
STOP_WORDS.update(EXTRA_STOP_WORDS)

lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Remove LaTeX, special characters, and normalize whitespace."""
    # Remove LaTeX math expressions
    text = re.sub(r"\$.*?\$", " ", text, flags=re.DOTALL)
    # Remove LaTeX commands
    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove punctuation (keep hyphens inside words)
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def tokenize_and_filter(text: str) -> List[str]:
    """Tokenize text, remove stop words, and lemmatize."""
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t.isalpha()               # keep only alphabetic tokens
        and len(t) > 2               # discard very short tokens
        and t not in STOP_WORDS      # remove stop words
    ]
    return tokens


def preprocess_spacy(text: str) -> List[str]:
    """Use spaCy for more accurate lemmatization and POS-based filtering."""
    cleaned = clean_text(text)
    doc = nlp(cleaned)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and token.is_alpha
        and len(token.text) > 2
        and token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}
        and token.lemma_ not in STOP_WORDS
    ]
    return tokens


def preprocess(text: str) -> List[str]:
    """
    Preprocess a single text string.
    Uses spaCy if available, otherwise falls back to NLTK.
    """
    if USE_SPACY:
        return preprocess_spacy(text)
    return tokenize_and_filter(text)


def preprocess_corpus(texts: List[str], use_spacy: bool = True) -> List[List[str]]:
    """
    Preprocess a list of texts.

    Args:
        texts: List of raw text strings.
        use_spacy: Whether to use spaCy (if available).

    Returns:
        List of tokenized, lemmatized word lists.
    """
    processed = []
    for text in texts:
        if use_spacy and USE_SPACY:
            tokens = preprocess_spacy(text)
        else:
            tokens = tokenize_and_filter(text)
        processed.append(tokens)
    return processed


def get_sentences(text: str) -> List[str]:
    """Split text into sentences for extractive summarization."""
    return sent_tokenize(text)


def tokens_to_string(tokens: List[str]) -> str:
    """Join tokens back into a string (for TF-IDF vectorizer)."""
    return " ".join(tokens)
