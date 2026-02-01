"""
LangGraph Orchestration Engine for Piracy Detection System

This module orchestrates the piracy detection pipeline using LangGraph.
It sequences existing detection modules without re-implementing ML logic.

Flow:
    Reference mode: Ingest → Store in FAISS → Done
    Query mode: Validate → Visual Match → Audio Match → Temporal Analysis 
                → Confidence Scoring → Decision Routing → Explanation → Output

Human decisions ALWAYS override AI.
LLM is used ONLY for explanation, never for scoring or decisions.
"""

from typing import TypedDict, Optional, List, Dict, Any, Literal, Tuple
from pathlib import Path
import pickle
import numpy as np

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None
    # Don't raise immediately - allow module to be imported for testing individual functions

# Import existing modules
try:
    from .frame_extractor import extract_frames
    from .preprocessor import preprocess_frame, batch_preprocess_frames
    from .embedding_extractor import EmbeddingExtractor, aggregate_embeddings
    from .fingerprint_index import VisualFingerprintIndex
    from .audio_extractor import extract_audio
    from .audio_embeddings import get_audio_embedding, audio_similarity, adjust_confidence
    from .visual_confidence import compute_visual_confidence
    from .decision_engine import DecisionEngine, DecisionThresholds
    from .multimodal_explainer import MultimodalPiracyExplainer
except ImportError:
    # Fallback for direct execution
    from frame_extractor import extract_frames
    from preprocessor import preprocess_frame, batch_preprocess_frames
    from embedding_extractor import EmbeddingExtractor, aggregate_embeddings
    from fingerprint_index import VisualFingerprintIndex
    from audio_extractor import extract_audio
    from audio_embeddings import get_audio_embedding, audio_similarity, adjust_confidence
    from visual_confidence import compute_visual_confidence
    from decision_engine import DecisionEngine, DecisionThresholds
    from multimodal_explainer import MultimodalPiracyExplainer


# ────────────────────────────────────────────
# STATE DEFINITION
# ────────────────────────────────────────────

class PiracyDetectionState(TypedDict, total=False):
    """
    State container for piracy detection pipeline.
    
    Fields are progressively populated as the pipeline executes.
    """
    # Input
    video_path: str
    mode: Literal["reference", "query"]  # reference = add to DB, query = detect
    reference_content_id: Optional[str]  # Used in reference mode
    
    # Paths and intermediate data
    frames_dir: str
    audio_path: str
    frame_paths: List[str]
    frame_timestamps: Dict[str, float]
    
    # Visual processing
    frame_embeddings: List[np.ndarray]
    aggregated_visual_embedding: np.ndarray
    visual_matches: List[tuple]  # (metadata, similarity_score)
    top_match_metadata: Optional[Dict[str, Any]]
    matched_content_id: Optional[str]
    visual_similarity: float
    
    # Audio processing
    audio_embedding: np.ndarray
    reference_audio_embedding: Optional[np.ndarray]
    reference_audio_path: Optional[str]
    audio_similarity_score: Optional[float]
    
    # Temporal analysis
    temporal_consistency_score: float
    coverage_ratio: float
    matched_timestamps: List[tuple]
    
    # Confidence and decision
    visual_confidence: float
    adjusted_confidence: float
    final_confidence: float
    decision: Literal["ignore", "manual_review", "auto_flag"]

    # Confidence weighting (optional override)
    confidence_weights: Dict[str, float]
    
    # Explanation (optional, post-decision)
    explanation: Optional[str]
    
    # Output
    result: Dict[str, Any]
    error: Optional[str]


# ────────────────────────────────────────────
# GLOBAL RESOURCES
# ────────────────────────────────────────────

# These should be initialized once and passed to the graph
VISUAL_INDEX: Optional[VisualFingerprintIndex] = None
EMBEDDING_EXTRACTOR: Optional[EmbeddingExtractor] = None
DECISION_ENGINE: Optional[DecisionEngine] = None
EXPLAINER: Optional[MultimodalPiracyExplainer] = None


def initialize_resources(
    index_path: Optional[str] = None,
    embedding_model: str = "resnet50",
    thresholds: Optional[DecisionThresholds] = None,
    anthropic_api_key: Optional[str] = None
):
    """
    Initialize global resources for the pipeline.
    
    Args:
        index_path: Path to FAISS index (loads if exists, creates if not)
        embedding_model: CNN model to use ("resnet50" or "mobilenet_v2")
        thresholds: Decision thresholds
        anthropic_api_key: API key for explainer (optional)
    """
    global VISUAL_INDEX, EMBEDDING_EXTRACTOR, DECISION_ENGINE, EXPLAINER
    
    # Initialize embedding extractor
    EMBEDDING_EXTRACTOR = EmbeddingExtractor(model_name=embedding_model)
    
    # Initialize or load FAISS index
    if index_path and Path(index_path + '.faiss').exists():
        metadata_path = Path(index_path + '.metadata.pkl')
        expected_dim = EMBEDDING_EXTRACTOR.embedding_dim
        stored_dim = None

        if metadata_path.exists():
            try:
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                stored_dim = data.get('embedding_dim')
            except Exception:
                stored_dim = None

        if stored_dim is not None and stored_dim != expected_dim:
            print(
                f"Warning: Existing index dimension {stored_dim} does not match "
                f"expected {expected_dim}. Creating new index."
            )
            VISUAL_INDEX = VisualFingerprintIndex(embedding_dim=expected_dim)
        else:
            try:
                VISUAL_INDEX = VisualFingerprintIndex(embedding_dim=expected_dim)
                VISUAL_INDEX.load(index_path)
            except Exception as e:
                # If loading fails (e.g., dimension mismatch), create new index
                print(f"Warning: Could not load existing index ({e}). Creating new index.")
                VISUAL_INDEX = VisualFingerprintIndex(embedding_dim=expected_dim)
    else:
        VISUAL_INDEX = VisualFingerprintIndex(embedding_dim=EMBEDDING_EXTRACTOR.embedding_dim)
    
    # Initialize decision engine
    DECISION_ENGINE = DecisionEngine(thresholds=thresholds)
    
    # Initialize explainer (optional)
    if anthropic_api_key:
        try:
            EXPLAINER = MultimodalPiracyExplainer(api_key=anthropic_api_key)
        except Exception:
            EXPLAINER = None


# ────────────────────────────────────────────
# NODE FUNCTIONS
# ────────────────────────────────────────────

def input_validation_node(state: PiracyDetectionState) -> PiracyDetectionState:
    """
    NODE 1: Validate input payload.
    
    Checks:
    - video_path exists
    - mode is valid
    - reference_content_id provided if mode == "reference"
    
    Fails fast if invalid.
    """
    try:
        # Validate video path
        video_path = Path(state["video_path"])
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {state['video_path']}")
        
        # Validate mode
        mode = state.get("mode", "query")
        if mode not in ["reference", "query"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'reference' or 'query'")
        
        # Validate reference mode requirements
        if mode == "reference" and not state.get("reference_content_id"):
            raise ValueError("reference_content_id required for reference mode")
        
        state["error"] = None
        return state
    
    except Exception as e:
        state["error"] = str(e)
        state["result"] = {"error": str(e), "decision": "error"}
        return state


def video_ingestion_node(state: PiracyDetectionState) -> PiracyDetectionState:
    """
    NODE 2: Ingest video and extract frames.
    
    For reference mode: Adds to FAISS index and ends.
    For query mode: Extracts frames for matching.
    """
    if state.get("error"):
        return state
    
    try:
        video_path = state["video_path"]
        mode = state["mode"]
        
        # Extract frames
        print(f"  → Extracting frames...")
        frames_dir = f"./data/frames/{Path(video_path).stem}"
        frame_metadata = extract_frames(video_path, frames_dir, fps=1)
        
        state["frames_dir"] = frames_dir
        state["frame_paths"] = list(frame_metadata.keys())
        state["frame_timestamps"] = frame_metadata
        print(f"  → Extracted {len(state['frame_paths'])} frames")
        
        # If reference mode, process and store immediately
        if mode == "reference":
            # Preprocess frames
            print(f"  → Preprocessing frames...")
            frame_tensors = batch_preprocess_frames(state["frame_paths"])
            
            # Generate embeddings
            print(f"  → Computing embeddings (this may take 1-2 minutes on CPU)...")
            embeddings_list = []
            for i, tensor in enumerate(frame_tensors):
                if i % 2 == 0:
                    print(f"     Processing frame {i+1}/{len(frame_tensors)}", end='\r')
                emb = EMBEDDING_EXTRACTOR.get_frame_embedding(tensor)
                embeddings_list.append(emb.numpy())
            print(f"  → Computed {len(embeddings_list)} embeddings" + " " * 20)

            # Extract reference audio (optional)
            reference_audio_embedding = None
            reference_audio_path = None
            try:
                reference_audio_path = f"./data/audio/{Path(video_path).stem}_ref.wav"
                extract_audio(video_path, reference_audio_path)
                reference_audio_embedding = get_audio_embedding(reference_audio_path)
            except Exception:
                reference_audio_embedding = None
                reference_audio_path = None

            state["reference_audio_embedding"] = reference_audio_embedding
            state["reference_audio_path"] = reference_audio_path
            
            # Add frame-level embeddings to FAISS index
            print(f"  → Adding to FAISS index...")
            content_id = state["reference_content_id"]
            timestamps = [state["frame_timestamps"].get(p) for p in state["frame_paths"]]
            additional_metadata = []
            for frame_path in state["frame_paths"]:
                additional_metadata.append({
                    "frame_path": frame_path,
                    "reference_audio_embedding": reference_audio_embedding,
                    "reference_audio_path": reference_audio_path,
                    "source": "reference_frame"
                })

            embeddings_array = np.stack(embeddings_list, axis=0)
            VISUAL_INDEX.add_fingerprints_batch(
                embeddings=embeddings_array,
                content_ids=[content_id] * len(embeddings_list),
                timestamps=timestamps,
                additional_metadata=additional_metadata
            )
            print(f"  ✓ Added {len(embeddings_list)} frame fingerprints to index")
            
            state["result"] = {
                "status": "reference_added",
                "content_id": state["reference_content_id"],
                "num_frames": len(state["frame_paths"]),
                "decision": "reference_stored"
            }
        
        return state
    
    except Exception as e:
        state["error"] = str(e)
        state["result"] = {"error": str(e), "decision": "error"}
        return state


def visual_matching_node(state: PiracyDetectionState) -> PiracyDetectionState:
    """
    NODE 3: Visual similarity matching using FAISS.
    
    Extracts visual embeddings and queries FAISS index.
    Outputs top matches with similarity scores.
    """
    if state.get("error") or state["mode"] == "reference":
        return state
    
    try:
        # Preprocess frames
        frame_tensors = batch_preprocess_frames(state["frame_paths"])

        # Generate embeddings for each frame
        embeddings_list = []
        for tensor in frame_tensors:
            emb = EMBEDDING_EXTRACTOR.get_frame_embedding(tensor)
            embeddings_list.append(emb.numpy())

        state["frame_embeddings"] = embeddings_list

        # Aggregate embeddings (video-level representation)
        aggregated = aggregate_embeddings(embeddings_list)
        state["aggregated_visual_embedding"] = aggregated

        # Frame-level matching against FAISS
        content_scores: Dict[str, List[float]] = {}
        content_matches: Dict[str, List[Tuple[Dict[str, Any], float]]] = {}
        matched_timestamps_by_content: Dict[str, List[tuple]] = {}

        for frame_path, frame_emb in zip(state["frame_paths"], embeddings_list):
            frame_matches = VISUAL_INDEX.search(frame_emb, top_k=1)
            if not frame_matches:
                continue

            top_metadata, top_similarity = frame_matches[0]
            content_id = top_metadata.get("content_id")
            if not content_id:
                continue

            content_scores.setdefault(content_id, []).append(top_similarity)
            content_matches.setdefault(content_id, []).append((top_metadata, top_similarity))

            query_ts = state["frame_timestamps"].get(frame_path)
            ref_ts = top_metadata.get("timestamp")
            matched_timestamps_by_content.setdefault(content_id, []).append((query_ts, ref_ts))

        if not content_scores:
            state["visual_similarity"] = 0.0
            state["visual_confidence"] = 0.0
            state["top_match_metadata"] = None
            state["matched_content_id"] = None
            state["visual_matches"] = []
            state["matched_timestamps"] = []
            return state

        # Select best content based on mean similarity
        best_content_id = max(content_scores.keys(), key=lambda cid: float(np.mean(content_scores[cid])))
        best_scores = content_scores[best_content_id]
        best_matches = content_matches[best_content_id]

        # Sort matches by similarity (desc)
        best_matches_sorted = sorted(best_matches, key=lambda x: x[1], reverse=True)

        state["matched_content_id"] = best_content_id
        state["visual_matches"] = best_matches_sorted
        state["top_match_metadata"] = best_matches_sorted[0][0]
        state["visual_similarity"] = float(np.mean(best_scores))
        state["matched_timestamps"] = matched_timestamps_by_content.get(best_content_id, [])

        # Compute visual confidence from best matches
        state["visual_confidence"] = compute_visual_confidence(best_matches_sorted, method="top1")

        return state
    
    except Exception as e:
        state["error"] = str(e)
        state["result"] = {"error": str(e), "decision": "error"}
        return state


def audio_matching_node(state: PiracyDetectionState) -> PiracyDetectionState:
    """
    NODE 4: Audio similarity matching.
    
    Extracts audio and compares against matched reference.
    Only runs if visual match found.
    """
    if state.get("error") or state["mode"] == "reference":
        return state
    
    try:
        # Skip if no visual match
        if state["visual_confidence"] < 0.3:
            state["audio_similarity_score"] = None
            return state
        
        # Extract audio from query video
        audio_path = f"./data/audio/{Path(state['video_path']).stem}.wav"
        extract_audio(state["video_path"], audio_path)
        state["audio_path"] = audio_path
        
        # Generate audio embedding
        audio_emb = get_audio_embedding(audio_path)
        state["audio_embedding"] = audio_emb

        # Get reference audio embedding from matched metadata (if available)
        reference_audio_embedding = None
        if state.get("top_match_metadata"):
            reference_audio_embedding = state["top_match_metadata"].get("reference_audio_embedding")

        state["reference_audio_embedding"] = reference_audio_embedding

        if reference_audio_embedding is not None:
            state["audio_similarity_score"] = audio_similarity(audio_emb, reference_audio_embedding)
        else:
            state["audio_similarity_score"] = None
        
        return state
    
    except Exception as e:
        # Audio is optional - don't fail if audio extraction fails
        state["audio_similarity_score"] = None
        return state


def temporal_coverage_analysis_node(state: PiracyDetectionState) -> PiracyDetectionState:
    """
    NODE 5: Temporal consistency and coverage analysis.
    
    Pure logic - no ML.
    Analyzes:
    - Temporal consistency (matches follow sequential order)
    - Coverage ratio (how much of video is matched)
    """
    if state.get("error") or state["mode"] == "reference":
        return state
    
    try:
        matched_timestamps = state.get("matched_timestamps", [])
        total_frames = len(state.get("frame_paths", []))
        matched_frames = len(matched_timestamps)

        # Coverage ratio based on matched frame count
        if total_frames > 0:
            coverage_ratio = matched_frames / total_frames
        else:
            coverage_ratio = 0.0

        # Temporal consistency: monotonic reference timestamps when ordered by query time
        temporal_consistency = 0.0
        if matched_frames >= 2:
            sorted_pairs = sorted(matched_timestamps, key=lambda x: (x[0] is None, x[0]))
            ref_times = [p[1] for p in sorted_pairs if p[1] is not None]

            if len(ref_times) >= 2:
                consistent_steps = 0
                for i in range(1, len(ref_times)):
                    if ref_times[i] >= ref_times[i - 1]:
                        consistent_steps += 1
                temporal_consistency = consistent_steps / (len(ref_times) - 1)
            else:
                temporal_consistency = 0.3
        elif matched_frames == 1:
            temporal_consistency = 0.2

        # Blend with coverage to penalize isolated matches
        temporal_consistency = (temporal_consistency * 0.7) + (coverage_ratio * 0.3)

        state["temporal_consistency_score"] = float(np.clip(temporal_consistency, 0.0, 1.0))
        state["coverage_ratio"] = float(np.clip(coverage_ratio, 0.0, 1.0))

        return state
    
    except Exception as e:
        state["error"] = str(e)
        state["result"] = {"error": str(e), "decision": "error"}
        return state


def confidence_scoring_node(state: PiracyDetectionState) -> PiracyDetectionState:
    """
    NODE 6: Compute final confidence score.
    
    This is the ONLY place confidence is computed.
    Uses weighted formula:
        final_confidence = w1*visual + w2*audio + w3*temporal + w4*coverage
    
    No LLM involvement - pure deterministic calculation.
    """
    if state.get("error") or state["mode"] == "reference":
        return state
    
    try:
        # Weights (configurable)
        weights = state.get("confidence_weights") or {
            "visual": 0.50,
            "audio": 0.20,
            "temporal": 0.15,
            "coverage": 0.15
        }

        w_visual = float(weights.get("visual", 0.50))
        w_audio = float(weights.get("audio", 0.20))
        w_temporal = float(weights.get("temporal", 0.15))
        w_coverage = float(weights.get("coverage", 0.15))

        weight_sum = w_visual + w_audio + w_temporal + w_coverage
        if weight_sum <= 0:
            weight_sum = 1.0
        if abs(weight_sum - 1.0) > 1e-6:
            w_visual /= weight_sum
            w_audio /= weight_sum
            w_temporal /= weight_sum
            w_coverage /= weight_sum
        
        # Get components
        visual = state["visual_confidence"]
        audio = state.get("audio_similarity_score", 0.0) or 0.0
        temporal = state["temporal_consistency_score"]
        coverage = state["coverage_ratio"]
        
        # Compute weighted confidence
        final_confidence = (
            w_visual * visual +
            w_audio * audio +
            w_temporal * temporal +
            w_coverage * coverage
        )
        
        # Clamp to [0, 1]
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        state["final_confidence"] = final_confidence
        
        # Also compute adjusted confidence (with audio boost if available)
        if audio > 0:
            state["adjusted_confidence"] = adjust_confidence(
                visual_confidence=visual,
                audio_similarity=audio
            )
        else:
            state["adjusted_confidence"] = visual
        
        return state
    
    except Exception as e:
        state["error"] = str(e)
        state["result"] = {"error": str(e), "decision": "error"}
        return state


def decision_router_node(state: PiracyDetectionState) -> PiracyDetectionState:
    """
    NODE 7: Route to appropriate action based on confidence.
    
    Routes to:
    - ignore: Low confidence, no action
    - manual_review: Borderline case, needs human review
    - auto_flag: High confidence, automatic flagging
    
    Uses deterministic thresholds from DecisionEngine.
    """
    if state.get("error") or state["mode"] == "reference":
        return state
    
    try:
        # Use final confidence score for routing
        decision = DECISION_ENGINE.make_decision(
            visual_confidence=state["final_confidence"],
            adjusted_confidence=None,
            use_adjusted=False
        )
        
        state["decision"] = decision
        
        return state
    
    except Exception as e:
        state["error"] = str(e)
        state["result"] = {"error": str(e), "decision": "error"}
        return state


def explanation_generator_node(state: PiracyDetectionState) -> PiracyDetectionState:
    """
    NODE 8: Generate human-readable explanation (OPTIONAL).
    
    Only triggered if:
    - Decision is auto_flag or manual_review
    - Explainer is available
    
    Uses READ-ONLY inputs - does NOT influence detection.
    LLM explains what was found, not what to decide.
    """
    if state.get("error") or state["mode"] == "reference":
        return state
    
    # Only explain if flagged or needs review
    if state["decision"] not in ["auto_flag", "manual_review"]:
        state["explanation"] = None
        return state
    
    # Skip if explainer not available
    if not EXPLAINER:
        state["explanation"] = None
        return state
    
    try:
        # Build payload for explainer
        payload = {
            "visual_similarity": state["visual_similarity"],
            "audio_similarity": state.get("audio_similarity_score"),
            "matched_content_id": state["top_match_metadata"]["content_id"] if state["top_match_metadata"] else "unknown",
            "query_video_id": Path(state["video_path"]).stem,
            "timestamps": state.get("matched_timestamps", []),
            "frame_paths": state["frame_paths"][:3],  # First 3 frames as evidence
            "metadata": {
                "temporal_consistency": state["temporal_consistency_score"],
                "coverage_ratio": state["coverage_ratio"]
            }
        }
        
        # Generate explanation (READ-ONLY operation)
        explanation = EXPLAINER.generate_explanation(payload)
        state["explanation"] = explanation
        
        return state
    
    except Exception:
        # Explanation is optional - don't fail pipeline if it errors
        state["explanation"] = None
        return state


def output_formatter_node(state: PiracyDetectionState) -> PiracyDetectionState:
    """
    NODE 9: Format final output.
    
    Returns structured result with:
    - decision
    - confidence_score
    - matched_content_id
    - explanation (if available)
    - review_required (bool)
    """
    if state.get("error"):
        return state
    
    # Reference mode already formatted
    if state["mode"] == "reference":
        return state
    
    try:
        result = {
            "decision": state["decision"],
            "confidence_score": state["final_confidence"],
            "visual_confidence": state["visual_confidence"],
            "adjusted_confidence": state["adjusted_confidence"],
            "matched_content_id": state.get("matched_content_id") or (state["top_match_metadata"]["content_id"] if state["top_match_metadata"] else None),
            "visual_similarity": state["visual_similarity"],
            "audio_similarity": state.get("audio_similarity_score"),
            "temporal_consistency": state["temporal_consistency_score"],
            "coverage_ratio": state["coverage_ratio"],
            "matched_timestamps": state.get("matched_timestamps", []),
            "review_required": state["decision"] == "manual_review",
            "auto_flagged": state["decision"] == "auto_flag",
            "explanation": state.get("explanation"),
            "num_frames_analyzed": len(state["frame_paths"]),
            "top_matches": [
                {"content_id": m[0]["content_id"], "similarity": m[1]}
                for m in state["visual_matches"][:3]
            ] if state["visual_matches"] else []
        }
        
        state["result"] = result
        
        return state
    
    except Exception as e:
        state["error"] = str(e)
        state["result"] = {"error": str(e), "decision": "error"}
        return state


# ────────────────────────────────────────────
# GRAPH CONSTRUCTION
# ────────────────────────────────────────────

def should_continue_after_validation(state: PiracyDetectionState) -> str:
    """Conditional edge: check if validation passed."""
    if state.get("error"):
        return "end"
    return "continue"


def should_continue_after_ingestion(state: PiracyDetectionState) -> str:
    """Conditional edge: reference mode ends here, query mode continues."""
    if state.get("error"):
        return "end"
    if state["mode"] == "reference":
        return "end"
    return "continue"


def should_skip_audio(state: PiracyDetectionState) -> str:
    """Conditional edge: skip audio if visual confidence too low."""
    if state.get("error"):
        return "skip"
    if state["visual_confidence"] < 0.3:
        return "skip"
    return "process"


def build_piracy_detection_graph() -> StateGraph:
    """
    Build LangGraph state machine for piracy detection.
    
    Returns:
        StateGraph ready for compilation
    """
    # Create graph
    workflow = StateGraph(PiracyDetectionState)
    
    # Add nodes
    workflow.add_node("input_validation", input_validation_node)
    workflow.add_node("video_ingestion", video_ingestion_node)
    workflow.add_node("visual_matching", visual_matching_node)
    workflow.add_node("audio_matching", audio_matching_node)
    workflow.add_node("temporal_coverage", temporal_coverage_analysis_node)
    workflow.add_node("confidence_scoring", confidence_scoring_node)
    workflow.add_node("decision_routing", decision_router_node)
    workflow.add_node("explanation", explanation_generator_node)
    workflow.add_node("output_formatter", output_formatter_node)
    
    # Set entry point
    workflow.set_entry_point("input_validation")
    
    # Add edges
    # Input validation → Video ingestion (if valid)
    workflow.add_conditional_edges(
        "input_validation",
        should_continue_after_validation,
        {
            "continue": "video_ingestion",
            "end": "output_formatter"
        }
    )
    
    # Video ingestion → Visual matching (query mode) or END (reference mode)
    workflow.add_conditional_edges(
        "video_ingestion",
        should_continue_after_ingestion,
        {
            "continue": "visual_matching",
            "end": END
        }
    )
    
    # Visual matching → Audio matching
    workflow.add_edge("visual_matching", "audio_matching")
    
    # Audio matching → Temporal coverage
    workflow.add_edge("audio_matching", "temporal_coverage")
    
    # Temporal coverage → Confidence scoring
    workflow.add_edge("temporal_coverage", "confidence_scoring")
    
    # Confidence scoring → Decision routing
    workflow.add_edge("confidence_scoring", "decision_routing")
    
    # Decision routing → Explanation
    workflow.add_edge("decision_routing", "explanation")
    
    # Explanation → Output formatter
    workflow.add_edge("explanation", "output_formatter")
    
    # Output formatter → END
    workflow.add_edge("output_formatter", END)
    
    return workflow


# ────────────────────────────────────────────
# ENTRY POINT
# ────────────────────────────────────────────

def run_piracy_detection(
    video_path: str,
    mode: Literal["reference", "query"] = "query",
    reference_content_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Main entry point for piracy detection pipeline.
    
    Args:
        video_path: Path to video file
        mode: "reference" to add to database, "query" to detect piracy
        reference_content_id: Content ID (required for reference mode)
        config: Optional configuration overrides
    
    Returns:
        Detection result dictionary
    
    Example:
        # Add reference video to database
        result = run_piracy_detection(
            "original_movie.mp4",
            mode="reference",
            reference_content_id="movie_xyz_2024"
        )
        
        # Check if a video is pirated
        result = run_piracy_detection(
            "suspicious_video.mp4",
            mode="query"
        )
    """
    # Check if LangGraph is available
    if not LANGGRAPH_AVAILABLE:
        raise ImportError(
            "langgraph not installed. Install with: pip install langgraph\n"
            "For testing individual nodes without LangGraph, use the node functions directly."
        )
    
    # Build and compile graph
    graph = build_piracy_detection_graph()
    app = graph.compile()
    
    # Initialize state
    initial_state: PiracyDetectionState = {
        "video_path": video_path,
        "mode": mode,
        "reference_content_id": reference_content_id
    }
    
    # Run pipeline
    final_state = app.invoke(initial_state, config=config)
    
    # Return result
    return final_state.get("result", {"error": "No result generated"})
