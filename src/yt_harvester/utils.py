"""Text and filesystem utilities for `yt_harvester`.

These helpers are designed to keep dataset artifacts consistent:
- Reliable extraction of YouTube video IDs from messy user input
- Caption cleanup that preserves Vietnamese diacritics
- Lightweight formatting utilities for human-readable reports
"""

import re
import html
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from typing import Iterable, List, Optional
from datetime import datetime

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
SENTENCE_ENDINGS = (".", "!", "?", "…")

def video_id_from_url(value: str) -> str:
    """Extract the 11-character YouTube video ID from a URL or raw ID.

    Why: users paste many different YouTube URL formats (watch/shorts/embed,
    youtu.be, with query params, etc.). This provides a single reliable parser.

    Args:
        value: A YouTube URL or a raw 11-character ID.

    Returns:
        The extracted 11-character video ID.

    Raises:
        ValueError: If no valid video ID can be extracted.
    """
    candidate = value.strip()
    if not candidate:
        raise ValueError("No video identifier provided.")

    if VIDEO_ID_RE.fullmatch(candidate):
        return candidate

    parsed = urlparse(candidate)
    host = (parsed.hostname or "").lower()

    if host in {"youtu.be", "www.youtu.be"}:
        parts = [segment for segment in parsed.path.split("/") if segment]
        if parts and VIDEO_ID_RE.fullmatch(parts[0]):
            return parts[0]

    if host.endswith("youtube.com"):
        query_params = parse_qs(parsed.query)
        if "v" in query_params:
            vid = query_params["v"][0]
            if VIDEO_ID_RE.fullmatch(vid):
                return vid
        path_segments = [segment for segment in parsed.path.split("/") if segment]
        if len(path_segments) >= 2 and path_segments[0] in {"embed", "shorts", "watch"}:
            vid = path_segments[1]
            if VIDEO_ID_RE.fullmatch(vid):
                return vid

    if "/" in candidate:
        tail = candidate.split("/")[-1]
        if VIDEO_ID_RE.fullmatch(tail):
            return tail

    raise ValueError("Unable to extract a valid YouTube video ID from the input.")


def build_watch_url(video_id: str) -> str:
    """Build a canonical YouTube watch URL.

    Args:
        video_id: YouTube video ID.

    Returns:
        Canonical watch URL (`https://www.youtube.com/watch?v=...`).
    """
    return f"https://www.youtube.com/watch?v={video_id}"


def cleanup_sidecar_files(video_id: str, suffixes: Iterable[str]) -> None:
    """Delete common `yt-dlp` sidecar files for a given video ID (best-effort).

    Why: `yt-dlp` writes `.info.json`, caption files, and other artifacts into the
    working directory. Cleaning these up avoids confusing leftovers between runs.

    Args:
        video_id: YouTube video ID.
        suffixes: Iterable of filename suffixes (e.g., `(".info.json", ".vtt")`).
    """
    for suffix in suffixes:
        candidate = Path(f"{video_id}{suffix}")
        if candidate.exists():
            try:
                candidate.unlink()
            except OSError:
                pass


def _strip_sentence_end(text: str) -> str:
    return text.rstrip('"\')]}»›”’')


def _is_sentence_end(text: str) -> bool:
    stripped = _strip_sentence_end(text)
    return bool(stripped) and stripped[-1] in SENTENCE_ENDINGS


def _normalise_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def merge_fragments(fragments: Iterable[str]) -> List[str]:
    """Merge caption fragments into sentence-like chunks.

    Why: caption formats often split a single sentence across many short timing
    fragments. Merging improves readability and produces cleaner transcript text
    for ASR training.

    Args:
        fragments: Iterable of raw caption fragments.

    Returns:
        A list of merged lines (roughly sentences).
    """
    paragraphs: List[str] = []
    buffer = ""
    for raw in fragments:
        text = _normalise_text(raw)
        if not text:
            continue
        buffer = f"{buffer} {text}".strip() if buffer else text
        if _is_sentence_end(buffer):
            if not paragraphs or paragraphs[-1] != buffer:
                paragraphs.append(buffer)
            buffer = ""
    if buffer:
        if not paragraphs or paragraphs[-1] != buffer:
            paragraphs.append(buffer)
    return paragraphs


def clean_caption_lines(path: Path) -> List[str]:
    """Clean caption files (VTT/SRT) into plain text lines.

    Why: raw captions include timestamps, headers, tags, and repeated lines.
    We strip those while keeping language content intact (including Vietnamese
    diacritics).

    Args:
        path: Path to a caption file.

    Returns:
        Cleaned caption lines. Returns an empty list if the file can't be read.
    """
    html_tag_re = re.compile(r"</?[^>]+>")
    inline_ts_re = re.compile(r"<\d{2}:\d{2}:\d{2}\.\d{3}>")
    cleaned: List[str] = []
    last_line = ""

    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                if line.upper() == "WEBVTT" or line.startswith("NOTE") or "-->" in line:
                    continue
                if path.suffix.lower() == ".srt" and line.isdigit():
                    continue
                line = html_tag_re.sub("", line)
                line = inline_ts_re.sub("", line)
                if line.startswith(("Kind:", "Language:", "Style:", "Region:")):
                    continue
                line = re.sub(r"\s+", " ", line).strip()
                if not line or line == last_line:
                    continue
                last_line = line
                cleaned.append(html.unescape(line))
    except OSError:
        return []
    return cleaned

def format_like_count(count: int) -> str:
    """Format like count to compact notation (e.g., 1.2M, 531k)."""
    if count >= 1_000_000:
        if count % 1_000_000 == 0:
            return f"{count // 1_000_000}M"
        else:
            formatted = f"{count / 1_000_000:.1f}M"
            return formatted.rstrip('0').rstrip('.')
    elif count >= 1_000:
        if count % 1_000 == 0:
            return f"{count // 1_000}k"
        else:
            formatted = f"{count / 1_000:.1f}k"
            return formatted.rstrip('0').rstrip('.')
    else:
        return str(count)

def format_timestamp(timestamp) -> str:
    """Format a timestamp into a human-readable date string.

    Args:
        timestamp: A UNIX timestamp (int/float) or an ISO-ish date string.

    Returns:
        Date formatted as `YYYY-MM-DD` when parseable, otherwise a best-effort
        string representation.
    """
    if not timestamp:
        return ""
    try:
        if isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(timestamp)
        else:
            # Try parsing ISO format or other formats
            dt = datetime.fromisoformat(str(timestamp).replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return str(timestamp) if timestamp else ""
