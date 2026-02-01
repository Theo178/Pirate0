# Multi-Project Repository

This repository contains multiple projects organized by functionality.

## Projects

### 📁 Anti-Piracy
Video piracy detection system using dual-modal fingerprinting (visual CNN + audio MFCC), FAISS vector search, and AI-powered forensic analysis.

**Tech Stack:** FastAPI, PyTorch, ResNet50, FAISS, Librosa, Google Gemini AI

**Documentation:** [Anti-Piracy README](./Anti-Piracy/README.md)

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

## Contributing

When adding new projects:
1. Create a dedicated folder at the root level
2. Include a comprehensive README.md in the project folder
3. Update this main README with project information
4. Maintain isolated dependencies per project
