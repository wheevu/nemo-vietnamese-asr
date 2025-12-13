import argparse
from .config import DEFAULT_CONFIG

def parse_args():
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
        "url",
        nargs="?",
        help="The URL or Video ID of the YouTube video (not used with --bulk)"
    )
    parser.add_argument(
        "-c", "--comments",
        type=int,
        default=DEFAULT_CONFIG["comments"]["top_n"],
        metavar="N",
        help=f"Number of top-level comments to fetch. Default: {DEFAULT_CONFIG['comments']['top_n']}"
    )
    parser.add_argument(
        "-f", "--format",
        choices=["txt", "json"],
        default=DEFAULT_CONFIG["output"]["format"],
        help=f"Output format for structured output. Default: {DEFAULT_CONFIG['output']['format']}"
    )
    parser.add_argument(
        "--max-comments",
        type=int,
        default=DEFAULT_CONFIG["comments"]["max_download"],
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
        help="Disable sentiment analysis."
    )
    parser.add_argument(
        "--no-keywords",
        action="store_true",
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
        default="./audio",
        help="Directory for audio files. Default: ./audio"
    )
    parser.add_argument(
        "--structured-dir",
        metavar="DIR",
        default="./structured_outputs",
        help="Directory for structured outputs. Default: ./structured_outputs"
    )
    parser.add_argument(
        "--transcripts-dir",
        metavar="DIR",
        default="./transcripts",
        help="Directory for raw transcripts. Default: ./transcripts"
    )
    
    return parser.parse_args()
