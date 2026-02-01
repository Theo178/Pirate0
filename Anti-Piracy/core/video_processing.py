"""
Video Processing - Reused from original with minimal changes
Extracts frames from videos using FFmpeg or OpenCV fallback
"""

import subprocess
from pathlib import Path
from typing import Dict
import os
def extract_frames(video_path: str, output_dir: str, fps: int = 1) -> Dict[str, float]:
    """
    Extract frames from video at specified FPS.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        fps: Frames per second to extract
    
    Returns:
        Dictionary mapping frame paths to timestamps
    """
    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Try FFmpeg first
    try:
        # Output pattern
        output_pattern = str(output_path / "frame_%06d.jpg")
        
        # FFmpeg command
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vf", f"fps={fps}",
            "-y",  # Overwrite
            output_pattern
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        
        # Collect frame files
        frame_files = sorted(output_path.glob("frame_*.jpg"))
        
        if not frame_files:
            raise RuntimeError("No frames extracted")
        
        # Build metadata
        frame_metadata = {}
        for idx, frame_file in enumerate(frame_files):
            timestamp = idx / fps
            frame_metadata[str(frame_file)] = timestamp
        
        return frame_metadata
        
    except FileNotFoundError:
        print("⚠ FFmpeg not found, using OpenCV fallback...")
        return extract_frames_opencv(video_path, output_dir, fps)


def extract_frames_opencv(video_path: str, output_dir: str, fps: int = 1) -> Dict[str, float]:
    """
    Extract frames using OpenCV (fallback when FFmpeg not available).
    """
    try:
        import cv2
    except ImportError:
        raise RuntimeError(
            "FFmpeg not found and OpenCV not installed. Install with: pip install opencv-python"
        )
    
    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        video_fps = 30  # Default fallback
    
    frame_interval = int(video_fps / fps)
    
    frame_metadata = {}
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame at specified interval
        if frame_count % frame_interval == 0:
            frame_filename = output_path / f"frame_{saved_count:06d}.jpg"
            cv2.imwrite(str(frame_filename), frame)
            timestamp = frame_count / video_fps
            frame_metadata[str(frame_filename)] = timestamp
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    
    if not frame_metadata:
        raise RuntimeError("No frames extracted")
    
    print(f"  ✓ Extracted {len(frame_metadata)} frames using OpenCV")
    return frame_metadata


def extract_audio(video_path: str, output_wav: str) -> str:
    """
    Extract audio from video as mono WAV.
    
    Args:
        video_path: Path to input video
        output_wav: Path to save WAV file
    
    Returns:
        Path to extracted audio file
    """
    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    output_path = Path(output_wav)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-q:a", "9",
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        "-y",  # Overwrite
        str(output_wav)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    
    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed: {result.stderr}")
    
    if not output_path.exists():
        raise RuntimeError("Audio file not created")
    
    return str(output_path)
