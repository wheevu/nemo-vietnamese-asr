"""Simple text analysis helpers for `yt_harvester`.

These are optional and meant for exploration (not required for ASR training).
They should never be treated as ground truth metrics.
"""

from typing import Dict, List
from textblob import TextBlob

def analyze_sentiment(text: str) -> Dict[str, float]:
    """Compute a lightweight sentiment score using TextBlob.

    Why: this is a quick heuristic that can help sort or review harvested
    content. It is **not** used for ASR training and should not be over-trusted.

    Args:
        text: Input text to analyze.

    Returns:
        A dict with:
        - `polarity`: float in [-1.0, 1.0]
        - `subjectivity`: float in [0.0, 1.0]
    """
    if not text:
        return {"polarity": 0.0, "subjectivity": 0.0}
    
    blob = TextBlob(text)
    return {
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity
    }

def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """Extract keyword phrases from text using TextBlob noun phrases.

    Why: keyword phrases can help you quickly understand what a video is about
    when browsing harvested data. This is optional and best-effort.

    Args:
        text: Input text to analyze.
        top_n: Maximum number of phrases to return.

    Returns:
        A list of keyword phrases sorted by frequency (highest first).
    """
    if not text:
        return []
    
    blob = TextBlob(text)
    # Get noun phrases and count frequency
    counts = blob.np_counts
    # Sort by frequency
    sorted_phrases = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [phrase for phrase, count in sorted_phrases[:top_n]]
