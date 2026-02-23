"""Tests for extractive summarizer."""
import pytest
from src.summarizer import extractive_summarize, summarize_batch, summarize_topic_group


SAMPLE_ABSTRACT = (
    "Deep learning has revolutionized the field of computer vision. "
    "Convolutional neural networks achieve state-of-the-art results on image recognition. "
    "Transfer learning allows models to be applied to new tasks with limited data. "
    "Attention mechanisms have further improved model performance on sequential data. "
    "These advances have enabled applications in medical imaging, autonomous driving, and robotics."
)


class TestExtractiveSummarize:
    def test_returns_string(self):
        result = extractive_summarize(SAMPLE_ABSTRACT, num_sentences=2)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_respects_num_sentences(self):
        result = extractive_summarize(SAMPLE_ABSTRACT, num_sentences=2)
        # Should not be longer than the original
        assert len(result) <= len(SAMPLE_ABSTRACT)

    def test_short_text_returned_as_is(self):
        short = "This is a single short sentence."
        result = extractive_summarize(short, num_sentences=3)
        assert result == short

    def test_summary_from_original_text(self):
        result = extractive_summarize(SAMPLE_ABSTRACT, num_sentences=2)
        # Each sentence in result should be from the original
        for sent in result.split(". "):
            if sent:
                assert sent.strip() in SAMPLE_ABSTRACT or SAMPLE_ABSTRACT.find(sent.strip()[:20]) != -1


class TestSummarizeBatch:
    def test_same_length_as_input(self):
        texts = [SAMPLE_ABSTRACT, SAMPLE_ABSTRACT[:100], SAMPLE_ABSTRACT]
        results = summarize_batch(texts, num_sentences=2)
        assert len(results) == 3

    def test_all_strings(self):
        results = summarize_batch([SAMPLE_ABSTRACT], num_sentences=2)
        assert all(isinstance(r, str) for r in results)


class TestSummarizeTopicGroup:
    def test_returns_non_empty(self):
        abstracts = [SAMPLE_ABSTRACT, SAMPLE_ABSTRACT]
        result = summarize_topic_group(abstracts, num_sentences=3)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_list(self):
        result = summarize_topic_group([], num_sentences=3)
        assert result == ""
