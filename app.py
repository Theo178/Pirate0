"""
Pirate0 v2.0 - Simplified FastAPI Server
Clean, simple, production-ready
"""

import os
# Fix OpenMP conflict between PyTorch and other libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import uuid
import asyncio
from typing import Optional

from core.detector import PiracyDetector

# Initialize FastAPI
app = FastAPI(
    title="Pirate0 v2.0 - Video Piracy Detection",
    description="Simplified CNN-based video fingerprinting",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for UI
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/frames", StaticFiles(directory="data/frames"), name="frames")

# Configuration
UPLOAD_DIR = Path("./data/raw")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Global detector instance
detector: Optional[PiracyDetector] = None


@app.on_event("startup")
async def startup_event():
    """Initialize detector on startup"""
    global detector
    try:
        print("Initializing Pirate0 v2.0...")
        detector = PiracyDetector(index_path="./data/index/video_index")
        print("✓ System ready")
    except Exception as e:
        print(f"✗ Failed to initialize detector: {e}")
        raise


@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Pirate0 v2.0",
        "version": "2.0.0",
        "description": "Simplified video piracy detection",
        "endpoints": {
            "GET /health": "Health check",
            "GET /stats": "Database statistics",
            "POST /reference": "Add reference video",
            "POST /query": "Check video for piracy",
            "DELETE /database": "Clear all fingerprints (start fresh)"
        }
    }


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "index_loaded": detector is not None and len(detector.index) > 0
    }


@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    if not detector:
        raise HTTPException(503, "Detector not initialized")
    
    stats = detector.get_stats()
    return {
        "status": "success",
        "database": stats
    }


@app.delete("/database")
async def clear_database():
    """
    Clear all fingerprints and data.
    Use this to start fresh.
    """
    if not detector:
        raise HTTPException(503, "Detector not initialized")
    
    try:
        result = detector.clear_database()
        return {
            "status": "success",
            "message": result['message'],
            "details": {
                "deleted_frames": result['deleted_frames'],
                "deleted_audio": result['deleted_audio']
            }
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to clear database: {str(e)}")


@app.post("/reference")
async def add_reference(
    video: UploadFile = File(...),
    content_id: str = Form(...),
    description: Optional[str] = Form(None)
):
    """
    Add reference video to database.
    
    Args:
        video: Video file
        content_id: Unique identifier (e.g., "movie_001")
        description: Optional description
    """
    # Validate detector
    if not detector:
        raise HTTPException(503, "Detector not initialized")
    
    # Validate
    if len(content_id) < 3:
        raise HTTPException(400, "content_id must be at least 3 characters")
    
    # Check file type
    if video.filename:
        ext = Path(video.filename).suffix.lower()
        if ext not in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            raise HTTPException(400, "Invalid video format")
    
    try:
        # Save uploaded file
        file_ext = Path(video.filename).suffix if video.filename else '.mp4'
        unique_filename = f"{content_id}_{uuid.uuid4().hex[:8]}{file_ext}"
        video_path = UPLOAD_DIR / unique_filename
        
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Process video
        result = detector.add_reference(str(video_path), content_id)
        
        return {
            "status": "success",
            "content_id": content_id,
            "num_frames": result['num_frames'],
            "has_audio_fingerprint": result.get('has_audio', False),
            "audio_error": result.get('audio_error'),
            "description": description
        }
    
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")


async def process_query_async(video_path: str) -> dict:
    """Process query in background (simulated async)"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, detector.check_video, video_path)


@app.post("/query")
async def check_piracy(
    video: UploadFile = File(...)
):
    """
    Check if video matches any reference content.
    
    Args:
        video: Video file to check
    """
    # Validate detector
    if not detector:
        raise HTTPException(503, "Detector not initialized")
    
    # Check file type
    if video.filename:
        ext = Path(video.filename).suffix.lower()
        if ext not in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            raise HTTPException(400, "Invalid video format")
    
    try:
        # Save uploaded file
        file_ext = Path(video.filename).suffix if video.filename else '.mp4'
        unique_filename = f"query_{uuid.uuid4().hex}{file_ext}"
        video_path = UPLOAD_DIR / unique_filename
        
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Process video (with async wrapper)
        result = await process_query_async(str(video_path))
        
        # Clean up
        video_path.unlink()
        
        # Format response
        return {
            "status": "success",
            "detection": {
                "decision": result['decision'],
                "confidence_score": round(result['confidence'], 3),
                "visual_confidence": round(result.get('visual_confidence', 0), 3),
                "audio_similarity": round(result['audio_similarity'], 3) if result.get('audio_similarity') is not None else None,
                "coverage_ratio": round(result.get('coverage_ratio', 0), 3),
                "temporal_consistency": round(result.get('temporal_consistency', 0), 3)
            },
            "matched_content": {
                "content_id": result.get('matched_content_id'),
                "num_frames_analyzed": result.get('num_frames', 0),
                "num_matches": result.get('num_matches', 0),
                "peak_similarity": round(result.get('peak_similarity', 0), 3),
                "avg_similarity": round(result.get('avg_similarity', 0), 3)
            },
            "evidence": result.get('evidence', {}),
            "ai_reasoning": result.get('ai_reasoning', 'AI reasoning not available'),
            "per_frame_analysis": result.get('per_frame_analysis', []),
            "action_required": {
                "review_required": result['decision'] == 'manual_review',
                "auto_flagged": result['decision'] == 'auto_flag'
            }
        }
    
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
