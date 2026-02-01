"""
Detection Engine - Simple and straightforward
No complex state machines, just a function
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from .video_processing import extract_frames
from .features import FeatureExtractor
from .index import VideoIndex
from .audio import extract_audio, get_audio_embedding, audio_similarity
from .ai_reasoning import EvidenceReasoner

# Minimum cosine similarity for a frame to count as a "match".
# ResNet50 features for unrelated images typically score 0.10-0.25.
FRAME_MATCH_THRESHOLD = 0.4


class PiracyDetector:
    """Simple piracy detection engine with audio-visual fingerprinting"""
    
    def __init__(self, index_path: str = "./data/index/video_index", enable_ai_reasoning: bool = True):
        """
        Initialize detector.
        
        Args:
            index_path: Path to FAISS index
            enable_ai_reasoning: Whether to enable AI-powered reasoning (requires GEMINI_API_KEY)
        """
        self.index_path = index_path
        self.audio_fingerprints_path = index_path + '_audio.pkl'
        self.feature_extractor = FeatureExtractor()
        self.index = VideoIndex(dimension=self.feature_extractor.embedding_dim)
        
        # Audio fingerprints storage {content_id: audio_embedding}
        self.audio_fingerprints = {}
        
        # AI reasoning (optional)
        self.ai_reasoner = EvidenceReasoner() if enable_ai_reasoning else None
        
        # Load existing index if available
        if Path(index_path + '.faiss').exists():
            try:
                self.index.load(index_path)
                print(f"✓ Loaded index with {len(self.index)} fingerprints")
            except Exception as e:
                print(f"⚠ Could not load index: {e}")
        
        # Load audio fingerprints if available
        if Path(self.audio_fingerprints_path).exists():
            try:
                import pickle
                with open(self.audio_fingerprints_path, 'rb') as f:
                    self.audio_fingerprints = pickle.load(f)
                print(f"✓ Loaded {len(self.audio_fingerprints)} audio fingerprints")

                # Validate audio fingerprint dimensions -- detect stale old-format embeddings
                expected_dim = 60  # n_mfcc(20) * 3 (mfcc + delta + delta2)
                stale_ids = [
                    cid for cid, emb in self.audio_fingerprints.items()
                    if hasattr(emb, 'shape') and emb.shape[0] != expected_dim
                ]
                if stale_ids:
                    print(f"⚠ Found {len(stale_ids)} audio fingerprints with wrong dimensions (expected {expected_dim}-dim).")
                    print(f"  Removing stale audio fingerprints: {stale_ids}")
                    print(f"  Re-index reference videos to regenerate audio fingerprints.")
                    for cid in stale_ids:
                        del self.audio_fingerprints[cid]
                    # Save cleaned fingerprints
                    with open(self.audio_fingerprints_path, 'wb') as f:
                        pickle.dump(self.audio_fingerprints, f)
            except Exception as e:
                print(f"⚠ Could not load audio fingerprints: {e}")
    
    def add_reference(self, video_path: str, content_id: str) -> Dict:
        """
        Add reference video to database (visual + audio).
        
        Args:
            video_path: Path to video file
            content_id: Unique identifier
        
        Returns:
            Result dictionary
        """
        print(f"Processing reference video: {content_id}")
        
        # Extract frames
        frames_dir = f"./data/frames/{content_id}"
        frame_metadata = extract_frames(video_path, frames_dir, fps=1)
        frame_paths = list(frame_metadata.keys())
        timestamps = list(frame_metadata.values())
        
        print(f"  Extracted {len(frame_paths)} frames")

        # Generate augmented visual embeddings (original + transformed variants)
        print(f"  Computing augmented visual embeddings...")
        aug_embeddings_list = self.feature_extractor.extract_batch_augmented(frame_paths)

        # Add each frame's augmented variants to the index.
        # All variants share the same content_id and timestamp.
        num_variants = 0
        for frame_augs, ts in zip(aug_embeddings_list, timestamps):
            aug_timestamps = [ts] * len(frame_augs)
            self.index.add(frame_augs, content_id, aug_timestamps)
            num_variants += len(frame_augs)

        print(f"  Indexed {num_variants} embeddings ({len(frame_paths)} frames x {num_variants // len(frame_paths)} variants)")
        
        # Extract and generate audio fingerprint
        print(f"  Extracting audio fingerprint...")
        audio_path = f"./data/audio/{content_id}.wav"
        audio_error = None
        try:
            extract_audio(video_path, audio_path)
            print(f"  [audio] WAV extracted to: {audio_path}")
            audio_embedding = get_audio_embedding(audio_path)
            if audio_embedding is not None:
                self.audio_fingerprints[content_id] = audio_embedding
                print(f"  ✓ Audio fingerprint generated (shape: {audio_embedding.shape})")
            else:
                audio_error = "MFCC embedding computation returned None (check server logs)"
                print(f"  [audio] get_audio_embedding returned None -- check logs above")
        except Exception as e:
            audio_error = str(e)
            print(f"  [audio] Audio extraction failed: {e}")

        # Save index
        self.index.save(self.index_path)

        # Save audio fingerprints
        import pickle
        with open(self.audio_fingerprints_path, 'wb') as f:
            pickle.dump(self.audio_fingerprints, f)

        print(f"✓ Added {content_id} to database")

        return {
            'status': 'success',
            'content_id': content_id,
            'num_frames': len(frame_paths),
            'has_audio': content_id in self.audio_fingerprints,
            'audio_error': audio_error,
        }
    
    def check_video(self, video_path: str) -> Dict:
        """
        Check if video matches any reference content (visual + audio).
        
        Args:
            video_path: Path to video file
        
        Returns:
            Detection result
        """
        if len(self.index) == 0:
            return {
                'decision': 'ignore',
                'confidence': 0.0,
                'message': 'No reference videos in database'
            }
        
        print("Analyzing video...")
        
        # Extract frames
        video_name = Path(video_path).stem
        frames_dir = f"./data/frames/{video_name}_query"
        frame_metadata = extract_frames(video_path, frames_dir, fps=1)
        frame_paths = list(frame_metadata.keys())
        
        print(f"  Extracted {len(frame_paths)} frames")
        
        # Generate visual embeddings
        print(f"  Computing visual embeddings...")
        embeddings = self.feature_extractor.extract_batch(frame_paths)
        
        # Search for each frame -- collect ALL matches for reporting,
        # but track which ones pass the threshold for scoring.
        print(f"  Searching for visual matches...")
        all_matches = []       # All frame matches (for grouping & evidence)
        strong_matches = []    # Above-threshold matches (for confidence scoring)
        frame_evidence = []

        for idx, (emb, frame_path) in enumerate(zip(embeddings, frame_paths)):
            matches = self.index.search(emb, top_k=1)
            if matches:
                metadata, score = matches[0]
                all_matches.append(matches[0])

                is_strong = score >= FRAME_MATCH_THRESHOLD
                if is_strong:
                    strong_matches.append(matches[0])

                # Use simple relative path from data/frames/
                try:
                    rel_path = str(Path(frame_path).relative_to(Path('./data/frames')))
                    serve_path = f"data/frames/{rel_path}"
                except ValueError:
                    serve_path = str(frame_path)

                frame_evidence.append({
                    'query_frame': Path(frame_path).name,
                    'query_frame_path': serve_path,
                    'query_frame_index': idx,
                    'matched_content_id': metadata['content_id'],
                    'similarity': float(score),
                    'reference_timestamp': float(metadata.get('timestamp', 0)),
                    'match_quality': 'high' if score > 0.85 else 'medium' if score > 0.6 else 'low' if score >= FRAME_MATCH_THRESHOLD else 'none'
                })

        # Group ALL matches by content_id to find the best reference
        content_all_scores = {}
        for metadata, score in all_matches:
            cid = metadata['content_id']
            if cid not in content_all_scores:
                content_all_scores[cid] = []
            content_all_scores[cid].append(score)

        # Find best matching reference (use max of mean scores across all frames)
        if content_all_scores:
            best_content_id = max(content_all_scores.keys(),
                                 key=lambda k: np.mean(content_all_scores[k]))
        else:
            best_content_id = None

        # Strong matches for the best reference (used for confidence scoring)
        strong_scores_for_best = [
            score for meta, score in strong_matches
            if meta['content_id'] == best_content_id
        ] if best_content_id else []

        # All scores for the best reference (used for reporting)
        all_scores_for_best = content_all_scores.get(best_content_id, [])

        # Visual confidence: mean of strong matches only (0.0 if none)
        visual_confidence = float(np.mean(strong_scores_for_best)) if strong_scores_for_best else 0.0

        # Coverage: fraction of query frames with a strong match
        coverage_ratio = len(strong_scores_for_best) / len(frame_paths) if len(frame_paths) > 0 else 0.0

        # Temporal consistency (uses all matches for ordering analysis)
        temporal_score = self._check_temporal_consistency(all_matches, best_content_id) if best_content_id else 0.0

        # Peak similarity: highest single-frame score (useful context for reports)
        peak_similarity = float(max(all_scores_for_best)) if all_scores_for_best else 0.0
        avg_similarity = float(np.mean(all_scores_for_best)) if all_scores_for_best else 0.0

        # Filter evidence for best match only
        best_frame_evidence = [e for e in frame_evidence if e.get('matched_content_id') == best_content_id]

        # Categorize matches by quality
        high_matches = [e for e in best_frame_evidence if e['match_quality'] == 'high']
        medium_matches = [e for e in best_frame_evidence if e['match_quality'] == 'medium']
        low_matches = [e for e in best_frame_evidence if e['match_quality'] == 'low']
        
        # Extract and compare audio (always attempt, even for weak visual matches)
        audio_score = None
        print(f"  Extracting audio fingerprint...")
        audio_path = f"./data/audio/{video_name}_query.wav"
        try:
            extract_audio(video_path, audio_path)
            query_audio_embedding = get_audio_embedding(audio_path)

            if query_audio_embedding is None:
                print(f"  [audio] Query audio embedding is None -- extraction or MFCC failed")
            elif not best_content_id:
                print(f"  [audio] No best_content_id from visual matching")
            elif best_content_id not in self.audio_fingerprints:
                print(f"  [audio] No stored audio fingerprint for '{best_content_id}'")
                print(f"  [audio] Stored fingerprints: {list(self.audio_fingerprints.keys())}")
            else:
                reference_audio_embedding = self.audio_fingerprints[best_content_id]
                print(f"  [audio] Query shape: {query_audio_embedding.shape}, Ref shape: {reference_audio_embedding.shape}")
                audio_score = audio_similarity(query_audio_embedding, reference_audio_embedding)
                if audio_score is not None:
                    print(f"  ✓ Audio similarity: {audio_score:.3f}")
                else:
                    print(f"  [audio] audio_similarity() returned None -- check shape mismatch")
        except Exception as e:
            print(f"  [audio] Audio processing failed: {e}")

        # Final confidence with audio
        if audio_score is not None:
            confidence = (
                0.45 * visual_confidence +
                0.20 * audio_score +
                0.20 * coverage_ratio +
                0.15 * temporal_score
            )
        else:
            confidence = (
                0.50 * visual_confidence +
                0.30 * coverage_ratio +
                0.20 * temporal_score
            )

        # Make decision
        if confidence < 0.35:
            decision = 'ignore'
        elif confidence < 0.75:
            decision = 'manual_review'
        else:
            decision = 'auto_flag'

        print(f"✓ Analysis complete")

        # Determine temporal pattern
        if temporal_score > 0.8:
            temporal_pattern = "sequential"
        elif temporal_score > 0.5:
            temporal_pattern = "mostly_sequential"
        else:
            temporal_pattern = "scattered"

        result = {
            'decision': decision,
            'confidence': float(confidence),
            'visual_confidence': float(visual_confidence),
            'audio_similarity': float(audio_score) if audio_score is not None else None,
            'coverage_ratio': float(coverage_ratio),
            'temporal_consistency': float(temporal_score),
            'matched_content_id': best_content_id,
            'num_frames': len(frame_paths),
            'num_matches': len(strong_scores_for_best),
            'peak_similarity': peak_similarity,
            'avg_similarity': avg_similarity,
            'evidence': {
                'matched_frames': best_frame_evidence[:10],
                'match_breakdown': {
                    'high_similarity': len(high_matches),
                    'medium_similarity': len(medium_matches),
                    'low_similarity': len(low_matches)
                },
                'temporal_pattern': temporal_pattern,
                'audio_match_quality': 'high' if audio_score and audio_score > 0.80 else
                                      'medium' if audio_score and audio_score > 0.5 else
                                      'low' if audio_score else None
            }
        }

        # Always generate AI reasoning
        if self.ai_reasoner and self.ai_reasoner.client:
            print(f"  Generating AI reasoning...")
            per_frame_reasoning = self.ai_reasoner.generate_per_frame_reasoning(result)
            if per_frame_reasoning:
                result['ai_reasoning'] = per_frame_reasoning['overall']
                result['per_frame_analysis'] = per_frame_reasoning['per_frame']
                print(f"  ✓ AI reasoning generated for {len(per_frame_reasoning['per_frame'])} frames")
            else:
                result['ai_reasoning'] = self.ai_reasoner.generate_quick_summary(result)
        elif self.ai_reasoner:
            result['ai_reasoning'] = self.ai_reasoner.generate_quick_summary(result)

        return result
    
    def _check_temporal_consistency(self, matches: List[Tuple], target_content_id: str) -> float:
        """Check if matched frames are in sequential order"""
        target_matches = [m for m in matches if m[0]['content_id'] == target_content_id]
        
        if len(target_matches) < 2:
            return 0.5  # Neutral score for single match
        
        # Extract timestamps
        timestamps = [m[0]['timestamp'] for m in target_matches]
        
        # Check monotonic increase
        consistent = 0
        for i in range(1, len(timestamps)):
            if timestamps[i] >= timestamps[i-1]:
                consistent += 1
        
        return consistent / (len(timestamps) - 1) if len(timestamps) > 1 else 0.5
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        return {
            'num_fingerprints': len(self.index),
            'num_audio_fingerprints': len(self.audio_fingerprints),
            'embedding_dimension': self.index.dimension
        }
    
    def clear_database(self) -> Dict:
        """Clear all fingerprints and start fresh"""
        import shutil
        
        # Reset in-memory structures
        self.index = VideoIndex(dimension=self.feature_extractor.embedding_dim)
        self.audio_fingerprints = {}
        
        # Delete persisted files
        try:
            if Path(self.index_path + '.faiss').exists():
                Path(self.index_path + '.faiss').unlink()
            if Path(self.index_path + '.metadata.pkl').exists():
                Path(self.index_path + '.metadata.pkl').unlink()
            if Path(self.audio_fingerprints_path).exists():
                Path(self.audio_fingerprints_path).unlink()
        except Exception as e:
            print(f"⚠️  Error deleting index files: {e}")
        
        # Optionally clear cached frames and audio
        frames_dir = Path("./data/frames")
        audio_dir = Path("./data/audio")
        
        deleted_frames = 0
        deleted_audio = 0
        
        try:
            if frames_dir.exists():
                for item in frames_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                        deleted_frames += 1
        except Exception as e:
            print(f"⚠️  Error clearing frames: {e}")
        
        try:
            if audio_dir.exists():
                for item in audio_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                        deleted_audio += 1
        except Exception as e:
            print(f"⚠️  Error clearing audio: {e}")
        
        return {
            'status': 'success',
            'message': 'Database cleared successfully',
            'deleted_frames': deleted_frames,
            'deleted_audio': deleted_audio
        }
