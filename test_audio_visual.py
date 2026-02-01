#!/usr/bin/env python3
"""
Quick test script for Pirate0 v2.0 with audio-visual fingerprinting
Tests with the same video to ensure high match
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.detector import PiracyDetector
from pathlib import Path

def test_audio_visual():
    print("=" * 60)
    print("Pirate0 v2.0 - Audio-Visual Fingerprinting Test")
    print("=" * 60)
    print()

    # Initialize detector
    print("Initializing detector...")
    detector = PiracyDetector(index_path="./data/index/test_index_av")
    print()

    # Use the same video for reference and query (has audio)
    test_video = Path("./data/raw/test_video.mp4")

    if not test_video.exists():
        # Try alternate location
        test_video = Path("../data/raw/test_video.mp4")

    if not test_video.exists():
        print("Test video not found:", test_video)
        print("Please ensure test_video.mp4 is in ./data/raw/")
        return

    # Add reference video
    print("Step 1: Adding reference video (with audio)")
    print("-" * 60)
    result1 = detector.add_reference(str(test_video), "test_content_001")
    print(f"Status: {result1['status']}")
    print(f"Frames: {result1['num_frames']}")
    print(f"Audio: {'Generated' if result1.get('has_audio') else 'Failed'}")
    print()

    print("Step 2: Checking same video for piracy (should match highly)")
    print("-" * 60)
    result2 = detector.check_video(str(test_video))
    print()

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Decision:             {result2['decision'].upper()}")
    print(f"Overall Confidence:   {result2['confidence']:.1%}")
    print(f"  Visual Confidence:  {result2['visual_confidence']:.1%}")
    if result2.get('audio_similarity') is not None:
        print(f"  Audio Similarity:   {result2['audio_similarity']:.1%}")
    else:
        print(f"  Audio Similarity:   N/A (no audio or comparison failed)")
    print(f"  Coverage Ratio:     {result2['coverage_ratio']:.1%}")
    print(f"  Temporal Score:     {result2['temporal_consistency']:.1%}")
    print(f"Matched Content:      {result2['matched_content_id']}")
    print()

    # Assertions for regression testing
    errors = []

    if result2['confidence'] < 0.85:
        errors.append(f"Same-video confidence too low: {result2['confidence']:.3f} (expected > 0.85)")

    if result2['visual_confidence'] < 0.90:
        errors.append(f"Same-video visual confidence too low: {result2['visual_confidence']:.3f} (expected > 0.90)")

    if result2['coverage_ratio'] < 0.80:
        errors.append(f"Same-video coverage too low: {result2['coverage_ratio']:.3f} (expected > 0.80)")

    if result2['decision'] != 'auto_flag':
        errors.append(f"Same-video decision should be 'auto_flag', got '{result2['decision']}'")

    if errors:
        print("ASSERTIONS FAILED:")
        for err in errors:
            print(f"  - {err}")
    else:
        print("All assertions passed!")

    # Interpretation
    if result2.get('audio_similarity') is not None and result2['audio_similarity'] > 0.9:
        print("Audio-visual fingerprinting WORKING!")
        print("   Both visual and audio fingerprints matched successfully.")
    elif result2.get('audio_similarity') is not None:
        print(f"Audio fingerprinting working but lower similarity: {result2['audio_similarity']:.1%}")
    else:
        print("Audio fingerprinting unavailable (visual-only mode)")
        print("   This could mean:")
        print("   - Video has no audio stream")
        print("   - Audio extraction failed")
        print("   - Audio embedding dimension mismatch (clear database and re-add)")

    print()
    print("=" * 60)

if __name__ == "__main__":
    test_audio_visual()
