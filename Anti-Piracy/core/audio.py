"""
Audio fingerprinting module - extracted and simplified from v1.0
Uses MFCC (Mel-Frequency Cepstral Coefficients) for audio similarity matching.
"""

import subprocess
import numpy as np
import librosa
from pathlib import Path
from typing import Optional


def extract_audio(video_path: str, output_wav_path: str) -> str:
    """
    Extract audio from video file using FFmpeg.
    
    Args:
        video_path: Path to input video file
        output_wav_path: Path to save extracted audio WAV file
    
    Returns:
        Path to extracted audio file
    
    Raises:
        FileNotFoundError: If video file doesn't exist
        RuntimeError: If FFmpeg extraction fails
    """
    # Validate video path
    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create output directory
    output_path_obj = Path(output_wav_path)
    output_path_obj.parent.mkdir(exist_ok=True, parents=True)
    
    # FFmpeg command to extract audio
    # -y: overwrite output file
    # -i: input file
    # -vn: no video
    # -ac 1: mono (1 channel)
    # -ar 16000: sample rate 16 kHz
    # -f wav: output format WAV
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite without asking
        "-i", str(video_path),
        "-vn",  # No video
        "-ac", "1",  # Mono
        "-ar", "16000",  # 16kHz
        "-f", "wav",  # WAV format
        "-loglevel", "error",  # Only show errors
        str(output_wav_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=False,       # raw bytes -- avoids UnicodeDecodeError on Windows
            check=False,
            timeout=60,       # prevent hangs on codec probing / stdin wait
            stdin=subprocess.DEVNULL,  # prevent FFmpeg from waiting on stdin
        )

        if result.returncode != 0:
            stderr_text = result.stderr.decode('utf-8', errors='replace')
            stderr_lower = stderr_text.lower()
            # Check if error is due to no audio stream
            if any(msg in stderr_lower for msg in [
                "does not contain any stream",
                "output file is empty",
                "no audio",
                "stream specifier",
                "encoder",
            ]):
                raise RuntimeError(f"Video has no audio stream: {video_path}")
            raise RuntimeError(f"FFmpeg audio extraction failed (rc={result.returncode}): {stderr_text[:500]}")

    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg not found. Install FFmpeg and ensure it is in your PATH."
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("FFmpeg audio extraction timed out after 60 seconds")

    # Verify output file was created and has content beyond WAV header
    if not output_path_obj.exists():
        raise RuntimeError(f"Audio extraction completed but no output file: {output_wav_path}")

    file_size = output_path_obj.stat().st_size
    if file_size < 1000:
        # WAV header is 44 bytes; a file under 1KB has essentially no audio data
        output_path_obj.unlink(missing_ok=True)
        raise RuntimeError(f"Extracted audio file is too small ({file_size} bytes) -- video may lack audio")

    return str(output_path_obj)


def get_audio_embedding(wav_path: str, n_mfcc: int = 20) -> Optional[np.ndarray]:
    """
    Generate audio fingerprint using MFCC features with delta and delta-delta.

    Args:
        wav_path: Path to input WAV file
        n_mfcc: Number of MFCC coefficients (default: 20)

    Returns:
        Audio embedding as NumPy array of shape (n_mfcc * 3,) including deltas
        Returns None if extraction fails
    """
    try:
        # Validate path
        wav_path_obj = Path(wav_path)
        if not wav_path_obj.exists():
            print(f"  [audio] File not found: {wav_path}")
            return None

        # Check file is non-trivial
        file_size = wav_path_obj.stat().st_size
        if file_size < 1000:
            print(f"  [audio] WAV file too small ({file_size} bytes): {wav_path}")
            return None

        # Load audio file (16kHz, mono)
        y, sr = librosa.load(str(wav_path), sr=16000, mono=True)

        # librosa.feature.delta requires at least `width` MFCC time frames
        # (default width=9).  With hop_length=512 at sr=16000, each frame is
        # ~32ms, so 9 frames ≈ 0.29s.  Require 0.5s for safe margin.
        min_duration = 0.5  # seconds
        if len(y) < sr * min_duration:
            print(f"  [audio] Audio too short ({len(y)/sr:.2f}s < {min_duration}s)")
            return None

        # Extract MFCC features -- shape (n_mfcc, time_steps)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        n_time_frames = mfcc.shape[1]
        if n_time_frames < 9:
            print(f"  [audio] Too few MFCC frames ({n_time_frames}) for delta computation")
            return None

        # Compute delta (velocity) and delta-delta (acceleration) features
        # These capture temporal dynamics and improve discrimination
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        # Average each across time, then concatenate
        # Final embedding: (n_mfcc * 3,) = (60,) for n_mfcc=20
        mfcc_mean = np.mean(mfcc, axis=1)
        delta_mean = np.mean(mfcc_delta, axis=1)
        delta2_mean = np.mean(mfcc_delta2, axis=1)

        embedding = np.concatenate([mfcc_mean, delta_mean, delta2_mean])

        print(f"  [audio] Embedding computed: {embedding.shape} from {n_time_frames} MFCC frames ({len(y)/sr:.1f}s audio)")
        return embedding

    except Exception as e:
        print(f"  [audio] Failed to extract audio embedding: {e}")
        return None


def audio_similarity(emb1: Optional[np.ndarray], emb2: Optional[np.ndarray]) -> Optional[float]:
    """
    Calculate cosine similarity between two audio embeddings.
    
    Args:
        emb1: First audio embedding
        emb2: Second audio embedding
    
    Returns:
        Similarity score in range [0, 1], where 1 is identical
        Returns None if either embedding is None or invalid
    """
    # Check for None
    if emb1 is None or emb2 is None:
        return None
    
    # Validate inputs
    if not isinstance(emb1, np.ndarray) or not isinstance(emb2, np.ndarray):
        return None
    
    if emb1.shape != emb2.shape:
        print(f"  [audio] Embedding shape mismatch: {emb1.shape} vs {emb2.shape}")
        print(f"  [audio] This usually means the database has old-format fingerprints.")
        print(f"  [audio] Clear the database and re-index reference videos to fix.")
        return None
    
    if emb1.size == 0 or emb2.size == 0:
        return None
    
    try:
        # Flatten to 1D
        emb1_flat = emb1.flatten()
        emb2_flat = emb2.flatten()
        
        # Compute norms
        norm1 = np.linalg.norm(emb1_flat)
        norm2 = np.linalg.norm(emb2_flat)
        
        # Handle zero vectors
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute cosine similarity: (A · B) / (||A|| ||B||)
        cosine_sim = np.dot(emb1_flat, emb2_flat) / (norm1 * norm2)

        # MFCC coefficients can be negative, so cosine sim spans [-1, 1].
        # Clamp to [0, 1]: negative correlation = completely dissimilar.
        similarity = float(np.clip(cosine_sim, 0.0, 1.0))

        return similarity
    
    except Exception as e:
        print(f"⚠️  Failed to compute audio similarity: {e}")
        return None
