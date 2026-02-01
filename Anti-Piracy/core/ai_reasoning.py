"""
AI-Powered Evidence Reasoning
Uses Google Gemini to generate human-readable explanations for detection decisions
"""

import os
from typing import Dict, Optional
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


SYSTEM_PROMPT = """You are a senior digital forensics analyst specializing in video piracy detection and content identification. You work for a content protection platform that uses dual-modal fingerprinting (visual CNN embeddings via ResNet50 + audio MFCC analysis) combined with FAISS vector similarity search.

Your role is to interpret raw detection metrics and translate them into clear, actionable forensic reports. You understand:
- Visual similarity: ResNet50 extracts 2048-dimensional embeddings per frame, compared via cosine similarity (raw, not rescaled). Scores >85% indicate near-identical visual content. 60-85% may indicate re-encoded, cropped, or color-shifted copies. Frames below 40% are filtered out as noise.
- Audio similarity: MFCC (Mel-Frequency Cepstral Coefficients) fingerprints capture tonal characteristics. High audio match + high visual match = very strong piracy signal. Audio mismatch with visual match may indicate dubbed/re-scored content.
- Coverage ratio: What fraction of query frames matched reference content. High coverage = full copy; low coverage = partial clip or coincidental similarity.
- Temporal consistency: Whether matched frames appear in the same sequential order as the reference. High temporal consistency = direct copy; low consistency with high similarity = re-edited/shuffled content.

Always be precise, cite the specific numbers, and clearly state your confidence level in the assessment."""

FRAME_ANALYSIS_PROMPT = """You are a digital forensics analyst examining individual frame matches from a video piracy detection system. The system uses ResNet50 CNN embeddings (2048-dim, cosine similarity) to compare frames.

Similarity interpretation guide (raw cosine similarity, not rescaled):
- 90-100%: Virtually identical frame. Likely a direct copy or lossless re-encode.
- 80-90%: Extremely high match. Minor compression artifacts or resolution differences.
- 65-80%: Strong match. Could be re-encoded, slightly cropped, or color-graded differently.
- 50-65%: Moderate match. Possible same scene with noticeable modifications (overlay, watermark, aspect ratio change).
- 40-50%: Weak match. Borderline -- could be similar content or heavy transformation.
- <40%: No match. Different content (filtered out by the system).

Provide a concise 2-sentence analysis: first describe what the similarity score technically means, then state what it implies about whether this frame is pirated content."""


class EvidenceReasoner:
    """Generate AI-powered explanations for piracy detection evidence"""

    def __init__(self):
        """Initialize Gemini client"""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("GEMINI_API_KEY not found in environment. AI reasoning will be disabled.")
            self.client = None
        else:
            self.client = genai.Client(api_key=api_key)

    def generate_reasoning(self, detection_result: Dict) -> Optional[str]:
        """
        Generate AI-powered reasoning for the detection result.

        Args:
            detection_result: Full detection result with evidence

        Returns:
            Human-readable explanation string, or None if API unavailable
        """
        if not self.client:
            return None

        try:
            context = self._build_context(detection_result)

            prompt = f"""{SYSTEM_PROMPT}

---

Analyze the following piracy detection scan results and produce a structured forensic report.

SCAN DATA:
{context}

Respond in this exact format (use these exact section headers):

VERDICT:
One clear sentence: Is this pirated content, suspicious, or clean? Include the overall confidence percentage.

VISUAL ANALYSIS:
2-3 sentences analyzing the visual fingerprint match. Reference the specific similarity percentages, the number of high/medium/low matches, and what this distribution pattern tells us. Note if the match quality suggests a direct copy, re-encode, cam-rip, or partial clip.

AUDIO ANALYSIS:
1-2 sentences on the audio fingerprint correlation. If audio data is available, explain whether it corroborates or contradicts the visual evidence. If unavailable, note that the assessment relies solely on visual evidence.

TEMPORAL PATTERN:
1-2 sentences on frame ordering. Does the temporal consistency suggest a sequential copy, a re-edited version, or isolated coincidental matches?

RISK ASSESSMENT:
One sentence giving a clear risk level (Critical / High / Moderate / Low / Negligible) with a brief justification tied to the combined evidence signals."""

            response = self.client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
                config={
                    "max_output_tokens": 1024,
                    "temperature": 0.2,
                },
            )

            reasoning = response.text.strip()
            return reasoning

        except Exception as e:
            print(f"AI reasoning failed: {e}")
            return None

    def generate_per_frame_reasoning(self, detection_result: Dict) -> Optional[Dict]:
        """
        Generate AI reasoning for each individual frame.

        Args:
            detection_result: Full detection result with evidence

        Returns:
            Dict with overall reasoning and per-frame analysis
        """
        if not self.client:
            return None

        try:
            evidence = detection_result.get('evidence', {})
            matched_frames = evidence.get('matched_frames', [])[:5]

            if not matched_frames:
                return None

            frame_analyses = []

            for frame in matched_frames:
                prompt = f"""{FRAME_ANALYSIS_PROMPT}

Frame: {frame['query_frame']}
Cosine Similarity: {frame['similarity']:.1%}
Match Quality Tier: {frame['match_quality']}
Matched Reference Timestamp: {frame['reference_timestamp']:.1f}s"""

                try:
                    response = self.client.models.generate_content(
                        model="gemini-3-flash-preview",
                        contents=prompt,
                        config={
                            "max_output_tokens": 150,
                            "temperature": 0.2,
                        },
                    )

                    frame_analyses.append({
                        'frame': frame['query_frame'],
                        'analysis': response.text.strip()
                    })
                except Exception as e:
                    print(f"Frame analysis failed: {e}")
                    frame_analyses.append({
                        'frame': frame['query_frame'],
                        'analysis': f"{frame['match_quality'].capitalize()} match ({frame['similarity']:.1%} cosine similarity) at reference timestamp {frame['reference_timestamp']:.1f}s."
                    })

            # Generate overall summary
            overall = self.generate_reasoning(detection_result) or self.generate_quick_summary(detection_result)

            return {
                'overall': overall,
                'per_frame': frame_analyses
            }

        except Exception as e:
            print(f"Per-frame reasoning failed: {e}")
            return None

    def _build_context(self, result: Dict) -> str:
        """Build context string from detection result"""
        confidence = result.get('confidence', 0)
        decision = result.get('decision', 'unknown')
        visual_conf = result.get('visual_confidence', 0)
        audio_sim = result.get('audio_similarity')
        coverage = result.get('coverage_ratio', 0)
        temporal = result.get('temporal_consistency', 0)
        evidence = result.get('evidence', {})

        decision_label = {
            'auto_flag': 'AUTO-FLAG (>=75% confidence)',
            'manual_review': 'MANUAL REVIEW (35-75% confidence)',
            'ignore': 'NO MATCH (<35% confidence)'
        }.get(decision, decision.upper())

        context = f"""System Decision: {decision_label}
Combined Confidence Score: {confidence:.1%}

--- Signal Breakdown ---
Visual Fingerprint Confidence: {visual_conf:.1%}
Audio Fingerprint Similarity: {f"{audio_sim:.1%}" if audio_sim is not None else "Not available (no audio extracted)"}
Frame Coverage Ratio: {coverage:.1%}
Temporal Consistency Score: {temporal:.1%}
"""

        if 'match_breakdown' in evidence:
            breakdown = evidence['match_breakdown']
            high = breakdown.get('high_similarity', 0)
            medium = breakdown.get('medium_similarity', 0)
            low = breakdown.get('low_similarity', 0)
            total = high + medium + low
            context += f"""
--- Match Quality Distribution ({total} total matched frames) ---
High Similarity (>90%): {high} frames
Medium Similarity (70-90%): {medium} frames
Low Similarity (<70%): {low} frames
"""

        if 'temporal_pattern' in evidence:
            context += f"Temporal Pattern Classification: {evidence['temporal_pattern']}\n"

        if 'audio_match_quality' in evidence and evidence['audio_match_quality']:
            context += f"Audio Match Quality Classification: {evidence['audio_match_quality']}\n"

        if 'matched_frames' in evidence and evidence['matched_frames']:
            context += "\n--- Individual Frame Matches (sample) ---\n"
            for i, frame in enumerate(evidence['matched_frames'][:5]):
                context += f"  Frame {i+1}: {frame['query_frame']} -> {frame['similarity']:.1%} similarity at ref timestamp {frame['reference_timestamp']:.1f}s [{frame['match_quality']}]\n"

        return context

    def generate_quick_summary(self, detection_result: Dict) -> str:
        """
        Generate a structured summary without API call.
        Fallback when Gemini is not available.
        """
        decision = detection_result.get('decision', 'unknown')
        confidence = detection_result.get('confidence', 0)
        visual_conf = detection_result.get('visual_confidence', 0)
        audio_sim = detection_result.get('audio_similarity')
        coverage = detection_result.get('coverage_ratio', 0)
        temporal = detection_result.get('temporal_consistency', 0)
        evidence = detection_result.get('evidence', {})
        breakdown = evidence.get('match_breakdown', {})

        high = breakdown.get('high_similarity', 0)
        medium = breakdown.get('medium_similarity', 0)
        low = breakdown.get('low_similarity', 0)
        total = high + medium + low

        if decision == 'auto_flag':
            verdict = f"VERDICT:\nPiracy detected with {confidence:.0%} confidence. This content is a likely unauthorized copy."
            visual = f"\nVISUAL ANALYSIS:\n{high} of {total} frames show >90% visual similarity, indicating a direct copy or minimal re-encoding."
            risk = "\nRISK ASSESSMENT:\nCritical -- strong multi-signal evidence of content piracy. Recommend immediate action."
        elif decision == 'manual_review':
            verdict = f"VERDICT:\nSuspicious content detected at {confidence:.0%} confidence. Manual review recommended."
            visual = f"\nVISUAL ANALYSIS:\n{high} high and {medium} medium similarity frames detected across {total} total matches. Pattern suggests possible re-encode or partial copy."
            risk = "\nRISK ASSESSMENT:\nModerate -- mixed signals require human review to confirm piracy."
        else:
            verdict = f"VERDICT:\nNo significant match found. Confidence is {confidence:.0%}, below the review threshold."
            visual = f"\nVISUAL ANALYSIS:\nOnly {high} of {total} frames show high similarity. The low match rate indicates this is likely different content."
            risk = "\nRISK ASSESSMENT:\nLow -- insufficient evidence to suggest content piracy."

        audio = f"\nAUDIO ANALYSIS:\n{'Audio similarity at ' + f'{audio_sim:.0%}' + ', ' + ('corroborating' if audio_sim and audio_sim > 0.5 else 'not corroborating') + ' the visual evidence.' if audio_sim is not None else 'No audio fingerprint available for comparison.'}"
        temporal_text = f"\nTEMPORAL PATTERN:\nTemporal consistency at {temporal:.0%}, {'suggesting sequential frame ordering consistent with a direct copy' if temporal > 0.7 else 'indicating non-sequential matches or re-edited content'}."

        return f"{verdict}{visual}{audio}{temporal_text}{risk}"
