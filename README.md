# Multi-Project Repository

This repository contains multiple projects organized by functionality.

## Overview

Production-grade implementations for content protection and analysis. The main idea is for platforms like Netflix to integrate these systems via API to have their Audio-Visual Fingerprints generated. Then platforms like WhatsApp, Telegram etc can integrate so that every user's MEDIA UPLOAD can be inspected, flagged for piracy and necessary action is taken.

## Projects

### 📁 Anti-Piracy
Video piracy detection system using dual-modal fingerprinting (visual CNN + audio MFCC), augmented reference indexing, FAISS vector search, and AI-powered forensic analysis via Google Gemini.

**Tech Stack:** FastAPI, PyTorch, ResNet50, FAISS, Librosa, Google Gemini AI

**Documentation:** [Anti-Piracy README](./Anti-Piracy/README.md)

**Key Features:**
- Dual-modal fingerprinting (visual CNN + audio MFCC)
- Augmented reference indexing (5 variants per frame)
- FAISS vector search with cosine similarity
- AI-powered forensic analysis via Google Gemini
- Production-ready FastAPI server

---

## Repository Structure

```
.
├── Anti-Piracy/          # Video piracy detection system
│   ├── app.py           # FastAPI server
│   ├── core/            # Core detection modules
│   ├── data/            # FAISS indices and processed data
│   ├── static/          # Web UI
│   └── requirements.txt # Python dependencies
│
└── (Additional projects will be added here)
```

## Getting Started

Each project has its own setup instructions. Navigate to the respective project folder and follow the README.

## Anti-Piracy Quick Start

### Confidence Scoring

**With audio available:**
```
confidence = 0.45 * visual + 0.20 * audio + 0.20 * coverage + 0.15 * temporal
```

**Without audio:**
```
confidence = 0.50 * visual + 0.30 * coverage + 0.20 * temporal
```

| Signal | Description |
|--------|-------------|
| Visual confidence | Mean cosine similarity of strong matches (>= 0.4 threshold) |
| Audio similarity | MFCC cosine similarity against best reference |
| Coverage ratio | Fraction of query frames with strong visual matches |
| Temporal consistency | Whether matched frames appear in sequential order |

### Decision Thresholds

| Confidence | Decision | Meaning |
|------------|----------|---------|
| < 0.35 | `ignore` | Different content |
| 0.35 - 0.75 | `manual_review` | Suspicious, needs human review |
| >= 0.75 | `auto_flag` | High confidence piracy match |

### AI Forensic Analysis
- **Provider:** Google Gemini (`gemini-3-flash-preview`)
- **Output:** Structured 5-section report (Verdict, Visual Analysis, Audio Analysis, Temporal Pattern, Risk Assessment)
- **Per-frame analysis:** Individual AI assessment for each matched frame
- **Fallback:** Deterministic structured summary when API is unavailable

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Visual embeddings | PyTorch + ResNet50 (ImageNet V2) |
| Audio fingerprinting | librosa MFCC (20 coeff + deltas) |
| Vector search | FAISS IndexFlatIP (cosine similarity) |
| API server | FastAPI + Uvicorn |
| Video processing | FFmpeg (frames + audio extraction) |
| AI reasoning | Google Gemini API |
| Frontend | Vanilla HTML/CSS/JS (dark theme) |

## Installation

```bash
# 1. Install FFmpeg
# Windows: choco install ffmpeg  OR  download from ffmpeg.org
# macOS:   brew install ffmpeg
# Linux:   sudo apt install ffmpeg

# 2. Create virtual environment
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy env.example .env
# Edit .env and set GEMINI_API_KEY for AI reasoning (optional)
```

## Usage

### Start the server
```bash
python app.py
# API:  http://localhost:8000
# UI:   http://localhost:8000/static/index.html
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info and endpoint list |
| `/health` | GET | Health check + index status |
| `/stats` | GET | Database statistics |
| `/reference` | POST | Add reference video (with augmented indexing) |
| `/query` | POST | Analyze suspicious video |
| `/database` | DELETE | Clear all fingerprints and start fresh |

### Add a reference video
```bash
curl -X POST http://localhost:8000/reference \
  -F "video=@original_movie.mp4" \
  -F "content_id=movie_001" \
  -F "description=Original theatrical release"
```

### Check a suspicious video
```bash
curl -X POST http://localhost:8000/query \
  -F "video=@suspicious.mp4"
```

### Clear the database
```bash
curl -X DELETE http://localhost:8000/database
```

## Project Structure

```
pirate0-v2/
├── core/
│   ├── detector.py           # PiracyDetector orchestrator (scoring, thresholds)
│   ├── features.py           # ResNet50 embeddings + augmentation pipeline
│   ├── audio.py              # MFCC fingerprinting (20 coeff + deltas)
│   ├── index.py              # FAISS IndexFlatIP wrapper
│   ├── video_processing.py   # FFmpeg frame + audio extraction
│   └── ai_reasoning.py       # Gemini AI forensic analysis
├── static/
│   └── index.html            # Full interactive UI
├── data/
│   ├── raw/                  # Uploaded video files
│   ├── frames/               # Extracted frames (by content_id)
│   ├── audio/                # Extracted WAV files
│   └── index/                # FAISS index + metadata + audio fingerprints
├── app.py                    # FastAPI server
├── test_audio_visual.py      # Validation test script
├── requirements.txt          # Python dependencies
├── env.example               # Environment template
└── DEMO.md                   # Presentation & demo walkthrough
```

## Performance Characteristics

| Operation | CPU | GPU |
|-----------|-----|-----|
| Frame extraction (FFmpeg) | ~1 FPS real-time | Same |
| ResNet50 embedding (per frame) | ~0.5s | ~0.05s |
| Augmented embedding (5 variants batched) | ~1.2s | ~0.08s |
| FAISS search (per frame) | < 10ms | Same |
| Audio MFCC extraction | ~2s for full video | Same |
| Gemini AI reasoning | ~3-5s | Same (API) |

## License

MIT

---

## Contributing

When adding new projects:
1. Create a dedicated folder at the root level
2. Include a comprehensive README.md in the project folder
3. Update this main README with project information
4. Maintain isolated dependencies per project
