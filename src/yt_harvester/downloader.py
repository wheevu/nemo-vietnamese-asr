import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from .utils import merge_fragments, clean_caption_lines, cleanup_sidecar_files

# Preferred transcript languages (will try in order, then fall back to any available)
PREFERRED_TRANSCRIPT_LANGS = ["vi", "en", "en-US", "en-GB", "en-CA", "en-AU"]

# Audio output directory for ASR-ready files
AUDIO_OUTPUT_DIR = Path("./audio")

# Type alias for structured comment data
CommentDict = dict  # {author, text, like_count, timestamp, id, replies}
StructuredComments = List[CommentDict]

def fetch_metadata(video_id: str, watch_url: str) -> dict:
    """Fetch video title and channel via yt-dlp; fall back to placeholders."""
    ydl_opts = {"quiet": True, "skip_download": True}
    info = {}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(watch_url, download=False)
    except Exception:
        info = {}

    title = info.get("title") if isinstance(info, dict) else None
    channel = info.get("uploader") if isinstance(info, dict) else None
    canonical = info.get("webpage_url") if isinstance(info, dict) else None
    
    # Extended metadata
    view_count = info.get("view_count") if isinstance(info, dict) else None
    duration = info.get("duration") if isinstance(info, dict) else None
    upload_date = info.get("upload_date") if isinstance(info, dict) else None
    description = info.get("description") if isinstance(info, dict) else None
    tags = info.get("tags") if isinstance(info, dict) else []

    return {
        "Title": title or "(Unknown title)",
        "Channel": channel or "(Unknown channel)",
        "URL": canonical or watch_url,
        "ViewCount": view_count,
        "Duration": duration,
        "UploadDate": upload_date,
        "Description": description,
        "Tags": tags
    }

def try_official_transcript(video_id: str) -> List[str]:
    """Try to fetch official (manual) transcript in any available language."""
    api = YouTubeTranscriptApi()
    
    # First, try preferred languages in order
    try:
        transcript = api.fetch(video_id, languages=PREFERRED_TRANSCRIPT_LANGS)
        return merge_fragments(chunk.text for chunk in transcript)
    except Exception:
        pass
    
    # Fall back to any available transcript
    try:
        transcript_list = api.list(video_id)
        # Filter for manual (non-generated) transcripts first
        manual_transcripts = [t for t in transcript_list if not t.is_generated]
        if manual_transcripts:
            transcript = manual_transcripts[0].fetch()
            return merge_fragments(chunk.text for chunk in transcript)
    except Exception:
        pass
    
    return []


def try_auto_captions(video_id: str, watch_url: str) -> List[str]:
    """
    Try to fetch auto-generated captions in the video's original language.
    Falls back to any available auto-caption if preferred languages aren't available.
    """
    output_pattern = f"{video_id}.%(ext)s"
    
    # Build language preference string: preferred languages first, then all others
    # Format: "vi,en,all" - tries Vietnamese, then English, then any available
    preferred_langs = ",".join(PREFERRED_TRANSCRIPT_LANGS) + ",all"
    
    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-auto-subs",
        "--sub-format",
        "vtt",
        "--sub-langs",
        preferred_langs,
        "--no-write-playlist-metafiles",
        "-o",
        output_pattern,
        watch_url,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        return ["(Transcript unavailable: yt-dlp is not installed.)"]
    except subprocess.CalledProcessError:
        cleanup_sidecar_files(video_id, (".info.json",))
        return []
    except Exception as exc:
        cleanup_sidecar_files(video_id, (".info.json",))
        return [f"(Transcript unavailable: {exc})"]

    caption_files = sorted(Path(".").glob(f"{video_id}*.vtt")) + sorted(Path(".").glob(f"{video_id}*.srt"))
    fragments: List[str] = []
    
    # Prioritize caption files by preferred language order
    selected_file = None
    for lang in PREFERRED_TRANSCRIPT_LANGS:
        for cf in caption_files:
            if f".{lang}." in cf.name or cf.name.endswith(f".{lang}.vtt") or cf.name.endswith(f".{lang}.srt"):
                selected_file = cf
                break
        if selected_file:
            break
    
    # Fall back to first available caption file
    if not selected_file and caption_files:
        selected_file = caption_files[0]
    
    if selected_file:
        fragments.extend(clean_caption_lines(selected_file))
    
    # Clean up all caption files
    for caption_file in caption_files:
        try:
            caption_file.unlink()
        except OSError:
            pass

    cleanup_sidecar_files(video_id, (".info.json",))
    if not fragments:
        return []
    return merge_fragments(fragments)


def try_auto_transcript_api(video_id: str) -> List[str]:
    """Try to fetch auto-generated transcript via youtube_transcript_api."""
    api = YouTubeTranscriptApi()
    
    try:
        transcript_list = api.list(video_id)
        # Look for auto-generated transcripts
        auto_transcripts = [t for t in transcript_list if t.is_generated]
        
        if auto_transcripts:
            # Prioritize by preferred languages
            for lang in PREFERRED_TRANSCRIPT_LANGS:
                for t in auto_transcripts:
                    if t.language_code.startswith(lang.split('-')[0]):
                        transcript = t.fetch()
                        return merge_fragments(chunk.text for chunk in transcript)
            
            # Fall back to first available auto-generated transcript
            transcript = auto_transcripts[0].fetch()
            return merge_fragments(chunk.text for chunk in transcript)
    except Exception:
        pass
    
    return []


def fetch_transcript(video_id: str, watch_url: str) -> List[str]:
    """
    Fetch transcript for a video, trying multiple sources:
    1. Official (manual) transcripts in preferred languages
    2. Official transcripts in any language
    3. Auto-generated transcripts via API
    4. Auto-captions via yt-dlp (fallback)
    """
    # Try official/manual transcripts first
    official = try_official_transcript(video_id)
    if official:
        return official

    # Try auto-generated transcripts via API
    auto_api = try_auto_transcript_api(video_id)
    if auto_api:
        return auto_api

    # Fall back to yt-dlp for auto-captions
    auto = try_auto_captions(video_id, watch_url)
    if auto:
        return auto

    return ["(Transcript unavailable.)"]


def fetch_comments(
    video_id: str, 
    watch_url: str, 
    max_dl: int = 10000, 
    top_n: int = 20
) -> StructuredComments:
    """
    Fetch comments via yt-dlp and return structured data. Does NOT clean up files - caller must do cleanup.
    """
    info_json_path = Path(f"{video_id}.info.json")
    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-comments",
        "--write-info-json",
        "--extractor-args",
        f"youtube:max_comments={max_dl};comment_sort=top",
        "--no-write-playlist-metafiles",
        "-o",
        f"{video_id}.%(ext)s",
        watch_url,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError, Exception):
        return []

    if not info_json_path.exists():
        return []

    try:
        with info_json_path.open("r", encoding="utf-8") as handle:
            info_data = json.load(handle)
            data = info_data.get("comments", [])
    except Exception:
        return []

    if not isinstance(data, list) or not data:
        return []

    children = defaultdict(list)
    roots = []
    for comment in data:
        parent_id = comment.get("parent")
        if parent_id and parent_id != "root":
            children[parent_id].append(comment)
        else:
            roots.append(comment)

    def normalise_likes(value) -> int:
        if isinstance(value, int):
            return max(value, 0)
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return 0

    # Sort root comments by like count (descending) and take top_n
    roots.sort(key=lambda c: normalise_likes(c.get("like_count")), reverse=True)
    top_roots = roots[:top_n]
    
    # Build structured comment data
    structured_comments = []
    for root in top_roots:
        root_replies = children.get(root.get("id"), [])
        replies_sorted = sorted(root_replies, key=lambda r: r.get("timestamp", 0), reverse=True)
        limited_replies = replies_sorted[:50]
        
        structured_comments.append({
            "author": root.get("author", ""),
            "text": root.get("text", ""),
            "like_count": normalise_likes(root.get("like_count")),
            "timestamp": root.get("timestamp"),
            "id": root.get("id"),
            "replies": [
                {
                    "author": reply.get("author", ""),
                    "text": reply.get("text", ""),
                    "like_count": normalise_likes(reply.get("like_count")),
                    "timestamp": reply.get("timestamp"),
                    "id": reply.get("id"),
                }
                for reply in limited_replies
            ]
        })
    
    return structured_comments


def download_audio(video_id: str, watch_url: str, output_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Download audio from a YouTube video and convert to 16kHz mono WAV format (ASR standard).
    
    Args:
        video_id: YouTube video ID
        watch_url: Full YouTube watch URL
        output_dir: Output directory for audio files (default: ./audio)
    
    Returns:
        Path to the saved WAV file, or None if download failed
    """
    audio_dir = output_dir or AUDIO_OUTPUT_DIR
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = audio_dir / f"{video_id}.wav"
    
    # Skip if already downloaded
    if output_path.exists():
        return output_path
    
    # yt-dlp options for audio extraction with ffmpeg post-processing
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio/best",
        "outtmpl": str(audio_dir / f"{video_id}.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        # FFmpeg postprocessor args to convert to 16kHz mono (ASR standard)
        "postprocessor_args": {
            "FFmpegExtractAudio": ["-ar", "16000", "-ac", "1"]
        },
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([watch_url])
        
        if output_path.exists():
            return output_path
        else:
            return None
    except Exception as exc:
        # Clean up any partial downloads
        for partial in audio_dir.glob(f"{video_id}.*"):
            try:
                partial.unlink()
            except OSError:
                pass
        return None
