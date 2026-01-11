# 🎙️ End-to-End Vietnamese ASR Pipeline with NVIDIA NeMo

[![NeMo](https://img.shields.io/badge/NVIDIA-NeMo-green)](https://github.com/NVIDIA/NeMo)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wheevu/nemo-vietnamese-asr/blob/main/NVIDIA_NeMo_ASR.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Run Tests](https://github.com/wheevu/nemo-vietnamese-asr/actions/workflows/run_tests.yml/badge.svg)](https://github.com/wheevu/nemo-vietnamese-asr/actions/workflows/run_tests.yml)

## 1. Executive Summary

This repository focuses on **data ingestion + preparation** for Vietnamese ASR and **GPU training / offline evaluation** with **NVIDIA NeMo**. It harvests unstructured YouTube audio/video, runs ETL to produce a clean speech corpus, and generates NeMo-compatible manifests for reproducible training.

Aligned with NVIDIA’s production mental model, **training and inference serving are separate systems**: training produces model artifacts; inference is delivered via a **GPU-accelerated inference endpoint** across a clear service boundary. This repo covers dataset creation, training, and training-time inference / offline evaluation; production inference serving (e.g., Riva / NIM-style microservices) is intentionally out of scope.

This separation-of-concerns framing is informed by the architectural vocabulary and serving mental model emphasized in **[NVIDIA’s NIM Microservices coursework](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-FX-23+V1).**

The design also supports a common constraint: develop on **Apple Silicon (M1)**, train on **Linux/CUDA** GPUs (dependencies, cross-platform paths, and data validation).

---

## 2. System Architecture

Hybrid **Local-to-Cloud** workflow optimized for reproducibility and cost.

![Architecture Diagram](./asset/mermaid-diagram.png)

> - **Local CPU ETL, Cloud GPU training:** Run cheap ETL locally (M1 CPU); reserve paid GPU time for training and **training-time inference / offline evaluation**.
> - **Transcript quality fallback:** Prefer manual transcripts over auto-generated sources to reduce noisy labels (“training pollution”).
> - **Segmented evaluation:** Chunk audio into 30s windows to avoid Conformer’s $O(N^2)$ attention memory blowups and prevent OOM on T4 GPUs.

### Production-Oriented View

- ASR inference is a **separate service**: a model sits behind an **inference endpoint**; this repo does not implement serving.
- Enforce a **service boundary**: data prep outputs versioned artifacts (audio, transcripts, manifests); training outputs versioned artifacts (checkpoints/configs) for downstream serving systems.
- Training artifacts are intentionally **runtime-agnostic**: produced for downstream serving, but not coupled to any specific serving runtime.
- **Offline data prep (local CPU):** `src/yt_harvester` handles ingestion + ETL, transcript fallback, and 16kHz mono WAV standardization.
- **GPU training (NeMo):** run training on Linux/CUDA GPUs; evaluation here is **training-time inference / offline evaluation** (batch transcription + WER).
- **Decoupled training and serving:** training optimizes for experimentation/throughput; serving optimizes for latency/concurrency/operations.
- **Client–service interaction (future):** integrate via a request/response contract (audio → transcript), aligning with a future inference service / serving layer.

### Scope & Non-Goals

- No production inference serving (no deployed ASR endpoint / online API).
- No container runtime or orchestration included.
- “Inference” here means **training-time inference / offline evaluation**, not serving.
- Goal: correct architecture (training ≠ serving) and forward-compatible artifacts for a GPU-backed serving stack.

---

## 3. 📂 Project Structure

```bash
nemo-vietnamese-asr/
├── src/
│   └── yt_harvester/           # 📦 Main Python package (Local ETL)
│       ├── __init__.py         # Public API exports
│       ├── __main__.py         # Entry point & orchestration
│       ├── cli.py              # CLI argument parsing
│       ├── config.py           # YAML configuration management
│       ├── downloader.py       # YouTube data fetching logic
│       ├── processor.py        # Text analysis (sentiment, keywords)
│       └── utils.py            # Helper functions
├── tests/                      # 🧪 pytest test suite
│   ├── conftest.py             # Shared fixtures (mock audio, manifests)
│   ├── test_text_processing.py # URL parsing, Vietnamese text normalization
│   └── test_data_integrity.py  # Manifest validation, audio compliance
├── prepare_data.py             # ⚙️ NeMo manifest generator & validator
├── benchmark.py                # 📊 Precision benchmarking (FP32/FP16/INT8)
├── audio/                      # Output: 16kHz mono WAV files
├── transcripts/                # Output: Clean raw transcript text
├── structured_outputs/         # Output: Full metadata + analysis
├── train_manifest.json         # NeMo training manifest
└── NVIDIA_NeMo_ASR.ipynb       # ☁️ Cloud Notebook (Training/Offline Evaluation)
```

---

## 4. 🛠️ Component 1: Local Data Engineering

The local engine (`src/yt_harvester`) covers **Extraction** + **Transformation**, turning raw YouTube videos into a structured speech dataset.

### Core Features

- **Smart Fallback Chain:** The `downloader.py` module attempts to fetch transcripts in the following order of quality:
  1.  Official Manual Transcripts (Vi/En)
  2.  Auto-Generated API Transcripts
  3.  Auto-Captions via `yt-dlp` CLI
- **Audio Standardization:** Converts streams to **16kHz mono WAV** via FFmpeg post-processing.
- **Idempotency:** Checks for existing files before downloading to save bandwidth and enable safe re-runs.
- **Rich Metadata:** Extracts sentiment polarity and top keywords using `TextBlob` for potential future downstream tasks.

### CLI Usage

```bash
# 1. Harvest a single video (Audio + Metadata + Transcript)
python -m src.yt_harvester "https://www.youtube.com/watch?v=VIDEO_ID"

# 2. Bulk harvest from a list of links
python -m src.yt_harvester --bulk links.txt --workers 4

# 3. Generate NeMo Manifests (Training/Validation/Test Split)
python prepare_data.py --seed 42
```

### Manifest Generation Strategy (`prepare_data.py`)

This script is the **validation layer** before training.

- **Integrity:** Cross-check `.wav` and `.txt`; discard missing/empty transcripts to prevent label noise.
- **Audio compliance:** Skip non-16kHz mono WAV files before training.
- **Normalization:** Lowercase + remove punctuation to match the CTC decoder alphabet.
- **Split:** Randomized 80/10/10 Train/Val/Test (use `--seed` for deterministic splits).

---

## 5. ☁️ Component 2: Cloud Workflow (Google Colab)

**File:** `NVIDIA_NeMo_ASR_Training.ipynb`

This component handles **model loading** plus **training-time inference / offline evaluation** on NVIDIA GPUs.

### Workflow Logic

1.  **Persistence:** mount Drive; unzip to local VM disk (`/content/data`) for fast I/O.
2.  **Model loading:** use NeMo’s polymorphic `ASRModel` to load `stt_en_conformer_ctc_large` from the **NVIDIA NGC Catalog** (avoids BPE vs char-class mismatch issues).
3.  **Path bridging:** remap macOS manifest paths (e.g., `/Users/josh/...`) to Colab paths (`/content/data/...`) at runtime.
4.  **Offline evaluation pipeline:** 30s chunking to prevent T4 OOM from Conformer’s $O(N^2)$ attention; decoding guarded for NeMo v2.6.0 signature changes (`paths2audio_files` vs positional args).

---

## 6. 📊 Results & Evaluation

Validation: **Zero-Shot Offline Evaluation (Training-Time Inference)** with a pre-trained English Conformer on the Vietnamese dataset.

- **Metric:** WER via `jiwer`.
- **Result:** WER ≈ 1.00 (expected for English-on-Vietnamese, zero-shot).
- **Qualitative:** Shows **phonetic mapping** (acoustic features are processed sensibly).

| Original Vietnamese Audio | Model Transcription (English Phonetics) | Analysis                       |
| :------------------------ | :-------------------------------------- | :----------------------------- |
| **"Giang Ơi Radio"**      | _"the radio"_                           | ✅ Recognized English loanword |
| **"Chào bạn"**            | _"ta bak"_                              | ✅ Acoustic approximation      |

**Conclusion:** The pipeline is ready for transfer learning (fine-tuning): freeze the encoder and retrain the decoder on Vietnamese.

---

## Production-Ready Model Optimization

This repo includes a production-oriented inference optimization workflow using **quantization** to reduce latency and memory footprint on **Google Colab T4** GPUs.

This benchmarking + quantization workflow is adapted from concepts practiced in the **[“Quantization Fundamentals with Hugging Face”](https://www.deeplearning.ai/short-courses/quantization-fundamentals-with-hugging-face/)** course, with an emphasis on the practical **accessibility gap**:
- **Why:** model sizes can exceed readily available VRAM (e.g., ~280 GB FP32 weights for a 70B model vs ~80 GB on A100 and ~16 GB on T4).
- **How:** reduce footprint by using lower-precision types (FP16 is ~2× smaller than FP32; INT8 is ~4× smaller than FP32), trading efficiency vs precision.

### Benchmarking (float32 vs float16 vs int8)

The benchmark script (`benchmark.py`) measures inference latency and WER across different precision settings. It automatically:
- Crops audio to 30s to avoid Conformer O(N²) OOM
- Remaps macOS manifest paths to Colab paths
- Handles NeMo API signature differences across versions
- Gracefully reports int8 failures (Quanto has known incompatibilities with NeMo Conformer)

**Run in Colab (Step 8B of the notebook):**
```bash
python benchmark.py \
  --model "/content/drive/MyDrive/Colab Notebooks/nemo_asr_project/vietnamese_asr_v1.nemo" \
  --manifest /content/data/test_manifest.json \
  --samples 100 \
  --audio-root /content/data/audio
```

**Or use an NGC pretrained model directly:**
```bash
python benchmark.py --model stt_en_conformer_ctc_large --manifest val_manifest.json --samples 100
```

### Results (Colab T4 GPU, Dec 2025)

| Precision | Avg Latency (ms/file) | WER (%) | VRAM (MB) | Samples | Notes |
| --- | --- | --- | --- | --- | --- |
| float32 | 151.54 | 99.89 | 731.29 | 2 | Baseline |
| float16 | 89.34 | 99.88 | 888.88 | 2 | **~40% faster**, recommended for T4 |
| int8 | — | — | 166.05 | 0 | Quanto incompatible with NeMo Conformer |

> **Note:** WER ~100% is expected — this is an **English** Conformer model (`stt_en_conformer_ctc_large`) evaluated on **Vietnamese** audio without fine-tuning. The benchmark validates the pipeline, not the model's Vietnamese accuracy.

---

## 7. 🧪 Testing

This project includes a professional test suite following best practices from **["Testing Machine Learning Systems: Code, Data and Models" by Made With ML](https://madewithml.com/courses/mlops/testing/)**. The tests validate both code correctness and data integrity — critical for ML pipelines where silent data issues cause training failures.

### Test Categories

| Category | File | Tests | Purpose |
| :------- | :--- | :---: | :------ |
| **Text Processing** | `test_text_processing.py` | 42 | YouTube URL parsing, Vietnamese text normalization, diacritic preservation |
| **Data Integrity** | `test_data_integrity.py` | 19 | NeMo manifest schema, 16kHz mono WAV compliance, file linkage |

### Key Testing Principles Applied

- **Fail Fast:** Catch data issues during preparation, not during expensive GPU training
- **Fixture-Based:** Reusable mock audio files and manifests via `conftest.py`
- **Negative Testing:** Verify validation catches invalid formats (stereo audio, wrong sample rate)
- **Vietnamese-Specific:** Tests ensure diacritics are preserved (e.g., "Việt" not corrupted to "Viet")

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Run specific test category
pytest tests/test_data_integrity.py -v
```

### Sample Test Output

```
tests/test_text_processing.py::TestVideoIdFromUrl::test_valid_url_formats[...] PASSED
tests/test_text_processing.py::TestCleanCaptionLines::test_vietnamese_diacritics_preserved PASSED
tests/test_data_integrity.py::TestAudioFormatCompliance::test_audio_sample_rate_is_16khz PASSED
tests/test_data_integrity.py::TestManifestAudioLinkage::test_manifest_audio_files_exist PASSED
========================= 61 passed in 0.34s =========================
```

---

## 8. 🚀 CI/CD (GitHub Actions)

This repo uses **GitHub Actions** to continuously validate code + data integrity on every change.

### Workflow Overview

- **Workflow file:** `.github/workflows/run_tests.yml`
- **Triggers:** `push` and `pull_request` to `main`
- **Runner:** `ubuntu-latest`
- **Python:** 3.10 (via `actions/setup-python`)

### What the Pipeline Does

1. **Checkout** the repository
2. **Install Linux system deps** (`libsndfile1`) so `soundfile` imports cleanly on Ubuntu
3. **Install Python dependencies** needed for the test suite and `src/` imports
4. **Run tests:**

```bash
pytest tests/ -v
```

### Notes

- This pipeline is **CI-focused** (test + validation). It does not deploy an inference service.

---

## 📦 Dependencies

- **Local:** `yt-dlp`, `ffmpeg`, `textblob`, `soundfile`, `pandas`
- **Cloud:** `nemo_toolkit[all]`, `pytorch-lightning`, `jiwer`, `librosa`
- **Testing:** `pytest`, `pytest-cov`, `numpy`, `soundfile`

> ### Conclusion & Next Steps
>
> Data processing is validated. WER ~1.00 matches the expected zero-shot failure mode (English model on Vietnamese). Next: **transfer learning**.
>
> **Proposed Fine-Tuning Strategy:**
>
> 1.  **Model Selection:** Utilize a smaller, pre-trained English model like `stt_en_conformer_ctc_small` from NGC.
> 2.  **Technique:** Freeze the audio **encoder** and fine-tune the language-specific **decoder** on the Vietnamese corpus.
> 3.  **Expectation:** Lower WER within a few epochs, demonstrating a practical path to Vietnamese ASR with modest compute.
>
> **Future Work:**
>
> - Develop a custom Vietnamese character-based tokenizer to replace the English BPE tokenizer for improved accuracy.
> - Perform detailed Error Analysis on the fine-tuned model to identify common phonetic failure points (e.g., tonal mistakes, loanwords).
