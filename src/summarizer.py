"""
Extractive Summarizer using word-frequency scoring and sentence ranking.
Implements a lightweight TextRank-inspired approach.
"""
from typing import List, Dict
import re
import math
import heapq
from collections import defaultdict

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))


def _sentence_scores(text: str) -> Dict[str, float]:
    """Score each sentence by the frequency of its significant words."""
    # Build word frequency table
    words = word_tokenize(text.lower())
    freq = defaultdict(float)
    for word in words:
        if word.isalpha() and word not in STOP_WORDS and len(word) > 2:
            freq[word] += 1.0

    # Normalize frequencies
    if freq:
        max_freq = max(freq.values())
        for word in freq:
            freq[word] /= max_freq

    # Score each sentence
    sentences = sent_tokenize(text)
    scores = {}
    for sent in sentences:
        sent_words = word_tokenize(sent.lower())
        score = sum(freq.get(w, 0.0) for w in sent_words if w.isalpha())
        # Normalize by sentence length to avoid bias toward long sentences
        length = max(len(sent_words), 1)
        scores[sent] = score / math.log(length + 1)

    return scores


def extractive_summarize(
    text: str,
    num_sentences: int = 3,
) -> str:
    """
    Generate a concise extractive summary of the input text.

    Args:
        text: The source text to summarize.
        num_sentences: Number of sentences to include in the summary.

    Returns:
        Summary string.
    """
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text  # Text is already short enough

    scores = _sentence_scores(text)
    # Pick top-scoring sentences
    top_sentences = heapq.nlargest(num_sentences, scores, key=scores.get)
    # Preserve original order
    ordered = [s for s in sentences if s in top_sentences]
    return " ".join(ordered)


def summarize_batch(
    texts: List[str],
    num_sentences: int = 3,
) -> List[str]:
    """
    Summarize a list of texts.

    Args:
        texts: List of abstract strings.
        num_sentences: Sentences per summary.

    Returns:
        List of summary strings.
    """
    return [extractive_summarize(t, num_sentences) for t in texts]


def summarize_topic_group(
    abstracts: List[str],
    num_sentences: int = 5,
) -> str:
    """
    Create a single collective summary from a group of abstracts
    belonging to the same topic/cluster.

    Args:
        abstracts: List of abstracts from papers in one topic.
        num_sentences: Sentences in the final summary.

    Returns:
        Merged extractive summary.
    """
    combined = " ".join(abstracts)
    return extractive_summarize(combined, num_sentences)
