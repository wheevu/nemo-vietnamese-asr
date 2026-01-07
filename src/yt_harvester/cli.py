"""CLI parsing for `yt_harvester`.

Config precedence (highest → lowest):
1) CLI flags
2) `config.yaml` (or `--config PATH`)
3) Built-in defaults in `DEFAULT_CONFIG`
"""

from __future__ import annotations

import argparse

from .config import DEFAULT_CONFIG, load_config

def parse_args():
    """Parse CLI args, using YAML config to seed defaults.

    Why two-phase parsing: we need to read `--config` first so we can use that
    YAML file to set argparse defaults for the rest of the flags.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(
        description="Harvest transcript, audio, and comments from YouTube videos for ASR training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output Structure:
  For each video, three files are created:
    ./audio/VIDEO_ID.wav           - 16kHz mono WAV audio (ASR-ready)
    ./structured_outputs/VIDEO_ID.txt - Full output (metadata, analysis, comments, transcript)
    ./transcripts/VIDEO_ID.txt     - Raw transcript text only

Examples:
  # Single video processing:
  python -m src.yt_harvester https://www.youtube.com/watch?v=VIDEO_ID
  python -m src.yt_harvester VIDEO_ID -c 10 -f json
  
  # Bulk processing:
  python -m src.yt_harvester --bulk links.txt
  python -m src.yt_harvester --bulk links.txt -f json --bulk-output-dir ./outputs
  
  # Skip audio download:
  python -m src.yt_harvester VIDEO_ID --no-audio
        """
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml).",
    )
    parser.add_argument(
        "url",
        nargs="?",
        help="The URL or Video ID of the YouTube video (not used with --bulk)"
    )

    # Two-phase parse: read --config early so YAML can seed argparse defaults.
    known, _ = parser.parse_known_args()
    cfg = load_config(getattr(known, "config", "config.yaml"))
    parser.add_argument(
        "-c", "--comments",
        type=int,
        default=int(cfg.get("comments", {}).get("top_n", DEFAULT_CONFIG["comments"]["top_n"])),
        metavar="N",
        help=f"Number of top-level comments to fetch. Default: {DEFAULT_CONFIG['comments']['top_n']}"
    )
    parser.add_argument(
        "-f", "--format",
        choices=["txt", "json"],
        default=str(cfg.get("output", {}).get("format", DEFAULT_CONFIG["output"]["format"])),
        help=f"Output format for structured output. Default: {DEFAULT_CONFIG['output']['format']}"
    )
    parser.add_argument(
        "--max-comments",
        type=int,
        default=int(cfg.get("comments", {}).get("max_download", DEFAULT_CONFIG["comments"]["max_download"])),
        metavar="N",
        help=f"Maximum total comments to download. Default: {DEFAULT_CONFIG['comments']['max_download']}"
    )
    parser.add_argument(
        "-o", "--output",
        metavar="FILE",
        help="Specify a custom output file name (structured output only)."
    )
    parser.add_argument(
        "--bulk",
        metavar="FILE",
        help="Process multiple videos from a file (one URL per line)."
    )
    parser.add_argument(
        "--bulk-output-dir",
        metavar="DIR",
        help="Base directory for bulk outputs. Creates audio/, structured_outputs/, transcripts/ subdirs."
    )
    parser.add_argument(
        "--no-sentiment",
        action="store_true",
        default=not bool(cfg.get("processing", {}).get("sentiment", DEFAULT_CONFIG["processing"]["sentiment"])),
        help="Disable sentiment analysis."
    )
    parser.add_argument(
        "--no-keywords",
        action="store_true",
        default=not bool(cfg.get("processing", {}).get("keywords", DEFAULT_CONFIG["processing"]["keywords"])),
        help="Disable keyword extraction."
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip audio download (audio is downloaded by default)."
    )
    parser.add_argument(
        "--audio-dir",
        metavar="DIR",
        default=str(cfg.get("output", {}).get("audio_dir", "./audio")),
        help="Directory for audio files. Default: ./audio"
    )
    parser.add_argument(
        "--structured-dir",
        metavar="DIR",
        default=str(cfg.get("output", {}).get("structured_dir", "./structured_outputs")),
        help="Directory for structured outputs. Default: ./structured_outputs"
    )
    parser.add_argument(
        "--transcripts-dir",
        metavar="DIR",
        default=str(cfg.get("output", {}).get("transcripts_dir", "./transcripts")),
        help="Directory for raw transcripts. Default: ./transcripts"
    )
    
    return parser.parse_args()
