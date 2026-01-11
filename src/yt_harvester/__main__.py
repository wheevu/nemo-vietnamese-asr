"""Orchestration entry point for `yt_harvester`.

This module keeps “what happens” obvious by splitting the pipeline into small
steps (metadata → transcript → audio → comments → analysis → save).
"""

from __future__ import annotations

import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from .cli import parse_args
from .downloader import download_audio, fetch_comments, fetch_metadata, fetch_transcript
from .processor import analyze_sentiment, extract_keywords
from .utils import (
    build_watch_url,
    cleanup_sidecar_files,
    format_like_count,
    format_timestamp,
    video_id_from_url,
)

# Default output directories
DEFAULT_AUDIO_DIR = Path("./audio")
DEFAULT_STRUCTURED_DIR = Path("./structured_outputs")
DEFAULT_TRANSCRIPTS_DIR = Path("./transcripts")

LOGGER = logging.getLogger(__name__)

YOUTUBE_HARVESTER_BANNER = r"""
________________________________________________________________________________
________________________________________________________________________________

__   __            _         _          _    _                           _
\ \ / /__  _   _  | |_ _   _| |__   ___| |  | | __ _ _ ____   _____  ___| |_ ___ _ __
 \ V / _ \| | | | | __| | | | '_ \ / _ \ |  | |/ _` | '__\ \ / / _ \/ __| __/ _ \ '__|
  | | (_) | |_| | | |_| |_| | |_) |  __/ |  | | (_| | |   \ V /  __/\__ \ ||  __/ |
  |_|\___/ \__,_|  \__|\__,_|_.__/ \___|_|  |_|\__,_|_|    \_/ \___||___/\__\___|_|

Welcome to YouTube Harvester!
________________________________________________________________________________
________________________________________________________________________________
""".strip("\n")

def format_comments_for_txt(structured_comments):
    """Format structured comments into readable text lines.

    This is used only for the `.txt` structured output to make nested replies
    easy to scan in a plain text file.

    Args:
        structured_comments: List of comment dictionaries. Each dict may include
            `author`, `text`, `like_count`, `timestamp`, and `replies`.

    Returns:
        A list of formatted lines (no trailing blank line).
    """
    if not structured_comments:
        return ["(No comments found.)"]
    
    def normalise_author(raw_author):
        if not raw_author:
            return "@Unknown"
        raw_author = raw_author.strip()
        return raw_author if raw_author.startswith("@") else f"@{raw_author}"
    
    def render_comment(comment_dict, depth=0):
        indent = "  " * depth
        arrow = "↳ " if depth else ""
        author = normalise_author(comment_dict.get("author"))
        likes = format_like_count(comment_dict.get("like_count", 0))
        text = (comment_dict.get("text") or "").replace("\n", " ").strip()
        
        # Add timestamp for root comments only
        if depth == 0:
            timestamp = comment_dict.get("timestamp")
            time_str = format_timestamp(timestamp)
            time_display = f" [{time_str}]" if time_str else ""
            line = f"{indent}{arrow}{author} (likes: {likes}){time_display}: {text or '(Comment deleted)'}"
        else:
            line = f"{indent}{arrow}{author} (likes: {likes}): {text or '(Comment deleted)'}"
        
        rendered_lines = [line]
        
        for reply in comment_dict.get("replies", []):
            rendered_lines.extend(render_comment(reply, depth + 1))
        
        return rendered_lines
    
    rendered_threads = []
    for root_comment in structured_comments:
        rendered_threads.extend(render_comment(root_comment))
        rendered_threads.append("")
    
    while rendered_threads and rendered_threads[-1] == "":
        rendered_threads.pop()
    
    return rendered_threads if rendered_threads else ["(No comments found.)"]

def save_txt(output_path, meta, transcript, comments, sentiment=None, keywords=None):
    """Write a human-readable structured report to disk.

    The text format is meant for quick inspection: metadata + optional analysis
    + transcript + comments.

    Args:
        output_path: Destination path for the `.txt` report.
        meta: Video metadata dictionary.
        transcript: Transcript lines (already cleaned/merged).
        comments: Formatted comment lines (already rendered for text output).
        sentiment: Optional sentiment payload from `analyze_sentiment`.
        keywords: Optional keyword list from `extract_keywords`.

    Returns:
        The same `output_path` for convenience.
    """
    transcript_lines = transcript or ["(Transcript unavailable.)"]
    comment_lines = comments or ["(Comments unavailable.)"]

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("====== METADATA ======\n")
        handle.write(f"Title: {meta.get('Title', '(Unknown title)')}\n")
        handle.write(f"Channel: {meta.get('Channel', '(Unknown channel)')}\n")
        handle.write(f"URL: {meta.get('URL', '')}\n")
        if meta.get("ViewCount"):
            handle.write(f"Views: {format_like_count(meta['ViewCount'])}\n")
        if meta.get("UploadDate"):
            handle.write(f"Uploaded: {meta['UploadDate']}\n")
        handle.write("\n")

        if sentiment or keywords:
            handle.write("====== ANALYSIS ======\n")
            if sentiment:
                handle.write(f"Sentiment: Polarity={sentiment['polarity']:.2f}, Subjectivity={sentiment['subjectivity']:.2f}\n")
            if keywords:
                handle.write(f"Keywords: {', '.join(keywords)}\n")
            handle.write("\n")

        handle.write("====== TRANSCRIPT ======\n")
        handle.write("\n\n".join(transcript_lines).strip() + "\n\n")

        handle.write("====== COMMENTS ======\n")
        handle.write("\n".join(comment_lines).strip() + "\n")

    return output_path

def save_json(output_path, meta, transcript, comments, sentiment=None, keywords=None):
    """Write a machine-readable structured report to disk (JSON).

    JSON is easier to post-process or load into downstream pipelines than the
    text format.

    Args:
        output_path: Destination path for the `.json` report.
        meta: Video metadata dictionary.
        transcript: Transcript lines (already cleaned/merged).
        comments: Structured comments (list of dicts with optional replies).
        sentiment: Optional sentiment payload from `analyze_sentiment`.
        keywords: Optional keyword list from `extract_keywords`.

    Returns:
        The same `output_path` for convenience.
    """
    full_data = {
        "metadata": meta,
        "analysis": {
            "sentiment": sentiment,
            "keywords": keywords
        },
        "transcript": transcript,
        "comments": comments
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(full_data, handle, indent=2, ensure_ascii=False)
    return output_path


def save_raw_transcript(output_path: Path, transcript: list) -> Path:
    """Write the raw transcript text to disk.

    This produces the `transcripts/VIDEO_ID.txt` file used for ASR training.
    We also filter out placeholder “(Transcript unavailable …)” lines so they
    don't pollute training labels.

    Args:
        output_path: Destination path for the transcript file.
        transcript: Transcript lines (strings).

    Returns:
        The saved transcript path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Filter out error messages like "(Transcript unavailable.)"
    clean_transcript = [
        line for line in transcript 
        if not line.startswith("(") or not line.endswith(")")
    ]
    
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(clean_transcript).strip())
    
    return output_path


def _resolve_output_dirs(args, bulk_base_dir: Optional[Path]) -> Tuple[Path, Path, Path]:
    """Compute output directories for this run.

    Why: bulk runs want a single base folder with consistent subfolders,
    while single-video runs respect the explicit `--audio-dir/--structured-dir/...`
    flags.

    Args:
        args: Parsed CLI args.
        bulk_base_dir: Optional base directory for bulk output.

    Returns:
        Tuple of `(audio_dir, structured_dir, transcripts_dir)`.
    """

    if bulk_base_dir:
        return (
            bulk_base_dir / "audio",
            bulk_base_dir / "structured_outputs",
            bulk_base_dir / "transcripts",
        )

    return (
        Path(getattr(args, "audio_dir", DEFAULT_AUDIO_DIR)),
        Path(getattr(args, "structured_dir", DEFAULT_STRUCTURED_DIR)),
        Path(getattr(args, "transcripts_dir", DEFAULT_TRANSCRIPTS_DIR)),
    )


def _ensure_dirs(*dirs: Path) -> None:
    """Create output directories if missing.

    Args:
        *dirs: One or more directory paths.
    """

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def _analyze_text(
    args, transcript_lines: Sequence[str]
) -> Tuple[Optional[Dict[str, float]], Optional[List[str]]]:
    """Run optional analysis (sentiment and/or keywords).

    Why: analysis is helpful for exploration, but it should never block the core
    ETL pipeline, so we keep this “best-effort” and optional via CLI flags.

    Args:
        args: Parsed CLI args (flags can disable analysis).
        transcript_lines: Transcript lines to analyze.

    Returns:
        `(sentiment, keywords)` where each value may be None when disabled.
    """

    full_text = " ".join(transcript_lines)
    sentiment = None if getattr(args, "no_sentiment", False) else analyze_sentiment(full_text)
    keywords = None if getattr(args, "no_keywords", False) else extract_keywords(full_text)
    return sentiment, keywords


def _save_outputs(
    *,
    args,
    video_id: str,
    structured_path: Path,
    transcript_path: Path,
    metadata: Dict[str, object],
    transcript: List[str],
    comments: List[dict],
    sentiment: Optional[Dict[str, float]],
    keywords: Optional[List[str]],
) -> None:
    """Write outputs to disk and clean up temporary sidecar files.

    Why cleanup exists: `yt-dlp` writes metadata and caption files into the
    current working directory. Removing them keeps re-runs tidy and prevents
    accidental mixing of artifacts across videos.

    Args:
        args: Parsed CLI args (controls output format).
        video_id: YouTube video ID (used for filenames and cleanup patterns).
        structured_path: Destination path for structured output (txt/json).
        transcript_path: Destination path for raw transcript output.
        metadata: Video metadata dictionary.
        transcript: Transcript lines.
        comments: Structured comment data.
        sentiment: Optional sentiment payload.
        keywords: Optional keyword list.
    """

    if getattr(args, "format", "txt") == "json":
        save_json(structured_path, metadata, transcript, comments, sentiment, keywords)
    else:
        formatted_comments = format_comments_for_txt(comments)
        save_txt(structured_path, metadata, transcript, formatted_comments, sentiment, keywords)

    save_raw_transcript(transcript_path, transcript)

    # yt-dlp writes sidecar files to the working directory; clean them up.
    cleanup_sidecar_files(
        video_id,
        (
            ".info.json",
            ".live_chat.json",
            ".vtt",
            ".srt",
            ".en.vtt",
            ".en-orig.vtt",
            ".en-en.vtt",
            ".en-de-DE.vtt",
        ),
    )
    for pattern in [f"{video_id}*.vtt", f"{video_id}*.srt"]:
        for file in Path(".").glob(pattern):
            try:
                file.unlink()
            except OSError:
                pass

def process_single_video(url, args, output_dir=None, pbar=None, progress_callback=None):
    """Harvest one video and write outputs.

    The pipeline is:
    metadata → transcript → (optional) audio → comments → (optional) analysis → save.

    Args:
        url: YouTube URL or raw 11-character video ID.
        args: Parsed CLI args.
        output_dir: Optional base output directory (used for bulk mode).
        pbar: Optional tqdm progress bar to update the description.
        progress_callback: Optional callback to update progress text.

    Returns:
        Tuple `(success, message)` where `message` is human-readable.
    """
    try:
        video_id = video_id_from_url(url)
    except ValueError as exc:
        return False, f"Invalid URL '{url}': {exc}"
    
    watch_url = build_watch_url(video_id)
    
    audio_dir, structured_dir, transcripts_dir = _resolve_output_dirs(args, output_dir)
    _ensure_dirs(audio_dir, structured_dir, transcripts_dir)
    
    # Output paths based on VIDEO_ID
    structured_output_path = structured_dir / f"{video_id}.{args.format}"
    transcript_output_path = transcripts_dir / f"{video_id}.txt"
    
    if pbar: pbar.set_description(f"Processing {video_id}")

    try:
        # 1) Metadata
        if progress_callback:
            progress_callback("Fetching metadata...")
        metadata = fetch_metadata(video_id, watch_url)
        
        # 2) Transcript
        if progress_callback:
            progress_callback("Fetching transcript...")
        transcript = fetch_transcript(video_id, watch_url)
        
        # 3) Audio (optional)
        audio_path = None
        if not getattr(args, "no_audio", False):
            if progress_callback:
                progress_callback("Downloading audio...")
            audio_path = download_audio(video_id, watch_url, output_dir=audio_dir)
        
        # 4) Comments
        if progress_callback:
            progress_callback("Fetching comments...")
        structured_comments = fetch_comments(
            video_id,
            watch_url,
            max_dl=args.max_comments,
            top_n=args.comments,
        )
        
        # 5) Analysis (optional)
        if progress_callback:
            progress_callback("Analyzing content...")
        sentiment, keywords = _analyze_text(args, transcript)

        # 6) Save outputs + cleanup
        if progress_callback:
            progress_callback("Saving outputs...")
        _save_outputs(
            args=args,
            video_id=video_id,
            structured_path=structured_output_path,
            transcript_path=transcript_output_path,
            metadata=metadata,
            transcript=transcript,
            comments=structured_comments,
            sentiment=sentiment,
            keywords=keywords,
        )
        
        # Build success message
        outputs = [f"📄 {structured_output_path}", f"📝 {transcript_output_path}"]
        if audio_path:
            outputs.insert(0, f"🔊 {audio_path}")
        
        msg = f"✅ {video_id}\n   " + "\n   ".join(outputs)
                    
        return True, msg

    except Exception as exc:
        return False, f"❌ {video_id}: {exc}"

def main():
    """CLI entry point.

    Returns:
        Process exit code (0 for success, non-zero for failure).
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    LOGGER.info(YOUTUBE_HARVESTER_BANNER)
    args = parse_args()
    # `parse_args()` already incorporates config defaults (CLI overrides config).

    if args.bulk:
        try:
            with open(args.bulk, "r", encoding="utf-8") as f:
                links = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        except Exception as e:
            LOGGER.error("❌ Error reading bulk file: %s", e)
            return 1

        if not links:
            LOGGER.warning("⚠️ No links found.")
            return 1

        output_dir = Path(args.bulk_output_dir) if args.bulk_output_dir else None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("🚀 Processing %s videos...", len(links))

        success_count = 0
        failed_count = 0

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            with tqdm(total=len(links), unit="video") as pbar:
                futures = {executor.submit(process_single_video, link, args, output_dir, pbar): link for link in links}

                for future in as_completed(futures):
                    success, msg = future.result()
                    if success:
                        success_count += 1
                    else:
                        failed_count += 1
                        tqdm.write(msg)
                    pbar.update(1)

        LOGGER.info("\n📊 Done! Success: %s, Failed: %s", success_count, failed_count)
        return 0 if failed_count == 0 else 1

    if not args.url:
        LOGGER.error("❌ No URL provided. Use --bulk or pass a video URL/ID.")
        return 1

    # Single video with detailed progress bar
    # Steps: metadata, transcript, [audio], comments, analysis, save
    # Audio is enabled by default, so 6 steps unless --no-audio is used
    total_steps = 5 if getattr(args, "no_audio", False) else 6
    with tqdm(total=total_steps, bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}]") as pbar:
        def update_progress(desc):
            pbar.set_description_str(desc)
            pbar.update(1)

        success, msg = process_single_video(args.url, args, progress_callback=update_progress)

        # Ensure bar completes if successful
        if success and pbar.n < total_steps:
            pbar.update(total_steps - pbar.n)

    LOGGER.info(msg)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
