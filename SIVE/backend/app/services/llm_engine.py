from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma  # Updated import
import json
import os

class LLMEngine:
    def __init__(self, model_name="mistral"):
        self.llm = OllamaLLM(model=model_name)
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.vector_store_path = "chroma_db"
        self.vector_store = None

    def initialize_vector_store(self, chunks: list[str]):
        # Initialize client with persistence
        self.vector_store = Chroma.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            persist_directory=self.vector_store_path
        )
        # Chroma 0.4+ persists automatically or uses distinct methods, 
        # but langchain-chroma handles logic. No .persist() method strictly needed in newer versions often.
        

    def get_context(self, query: str, k=2):
        try:
            if not self.vector_store:
                 # Try to load existing
                if os.path.exists(self.vector_store_path):
                    self.vector_store = Chroma(persist_directory=self.vector_store_path, embedding_function=self.embeddings)
                else:
                    return ""
            
            docs = self.vector_store.similarity_search(query, k=k)
            return "\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"Vector Store Retrieval Warning: {e}")
            return ""

    def analyze_script_context(self, full_text: str):
        """Extracts global context: Characters, Mood, Theme"""
        # We might not be able to process the FULL text if it's too long in one go.
        # Strategically we summarize. For now, let's take a substring or assume the user asks for query.
        # Better approach: The generic context is built via RAG or a summarization chain.
        # For simplicity in this turn, we'll ask for a summary of the first 5000 chars.
        
        snippet = full_text[:3000] # Further Reduced to save memory
        
        prompt = PromptTemplate(
            template="""You are a creative director's assistant. Analyze the beginning of this script to identify the main characters, the genre, and the overall visual tone.
            
            Script Excerpt:
            {text}
            
            Return a JSON object with:
            - "characters": list of names and brief descriptions
            - "genre": string
            - "visual_style": string description
            """,
            input_variables=["text"]
        )
        
        chain = prompt | self.llm | JsonOutputParser()
        try:
            return chain.invoke({"text": snippet})
        except Exception as e:
            print(f"LLM Context Analysis failed: {e}")
            return {
                "characters": [],
                "genre": "Unknown",
                "visual_style": "Standard cinematic style (Context extraction failed)"
            }

    def analyze_scene(self, scene_text: str):
        """Generates visual planning signals for a specific scene."""
        
        prompt_template_str = """You are an expert film director and cinematographer. Analyze the scene script below and extract detailed filmmaking data across 6 layers.

            Context:
            {context}

            Scene Script:
            {scene}

            Output strictly a valid JSON object with this exact structure:
            {{
                "scene_intent": {{
                    "emotion": "Primary emotion driving the scene",
                    "story_purpose": "Why this scene exists (e.g. power confrontation)",
                    "energy_level": "e.g. slow burn, high octane",
                    "audience_feeling_target": "What the audience should feel"
                }},
                "visual_mood": {{
                    "lighting_style": "e.g. low-key, high-key",
                    "contrast": "e.g. high, soft",
                    "color_palette": "e.g. cool desaturated, neon",
                    "shadow_density": "e.g. heavy, lifted"
                }},
                "camera_language": {{
                    "shot_plan": [
                        {{ "shot": "e.g. wide", "reason": "establish isolation" }},
                        {{ "shot": "e.g. close-up", "reason": "capture fear" }}
                    ],
                    "camera_motion": "e.g. slow push-in, static",
                    "lens_style": "e.g. compressed depth, wide angle"
                }},
                "actor_blocking": [
                    {{
                        "character": "Name",
                        "position": "e.g. center frame",
                        "posture": "e.g. tense shoulders",
                        "eye_focus": "e.g. locked stare",
                        "movement": "e.g. minimal controlled"
                    }}
                ],
                "editing_rhythm": {{
                    "pacing": "e.g. slow, frenetic",
                    "average_shot_length": "e.g. 7 seconds",
                    "cut_style": "e.g. hard cuts, dissolves",
                    "pause_usage": "e.g. dramatic silence emphasis"
                }},
                "production_logistics": {{
                    "complexity": "e.g. medium",
                    "shoot_time_estimate": "e.g. 4 hours",
                    "equipment": "e.g. dolly + soft light rig",
                    "risk_factor": "e.g. low"
                }}
            }}
            """
        
        prompt = PromptTemplate(
            template=prompt_template_str,
            input_variables=["context", "scene"]
        )
        
        chain = prompt | self.llm | JsonOutputParser()

        # Retry Logic for CUDA OOM / Memory Issues
        try:
            # Attempt 1: Light Context (k=1)
            context = self.get_context(scene_text, k=1)
            return chain.invoke({"context": context, "scene": scene_text})
        except Exception as e:
            print(f"Standard Analysis failed (likely Memory/CUDA): {e}")
            print("Retrying with ZERO context to save memory...")
            try:
                # Attempt 2: No Context (Memory Safe Mode)
                return chain.invoke({"context": "Context unavailable due to memory constraints.", "scene": scene_text})
            except Exception as e2:
                print(f"Critical Analysis Failure: {e2}")
                raise e2
    def analyze_character_arc(self, scene_text: str):
        """Extracts sequential emotional data for character arcs."""
        
        prompt = PromptTemplate(
            template="""You are a dramatic analyst. Analyze the following scene script chronologically.
            For each distinct line of dialogue or significant action, identify:
            1. The Character Name
            2. A brief snippet of the line/action (max 5 words)
            3. The primary Emotion (e.g., Angry, sarcastic, hesitant)
            4. Emotional Intensity (1-10, where 10 is explosive)
            5. Sentiment Score (-10 to 10, where -10 is extremely negative/hostile, 10 is pure joy/love)
            
            Scene Script:
            {scene}
            
            Output strictly a VALID JSON ARRAY of objects. Example:
            [
                {{ "character": "MARK", "line": "Did you know...", "emotion": "Condescending", "intensity": 4, "sentiment": -2 }},
                {{ "character": "ERICA", "line": "That can't be...", "emotion": "Skeptical", "intensity": 3, "sentiment": 0 }}
            ]
            """,
            input_variables=["scene"]
        )
        
        chain = prompt | self.llm | JsonOutputParser()
        try:
            return chain.invoke({"scene": scene_text})
        except Exception as e:
            print(f"Character Arc Analysis failed: {e}")
            return []

    def generate_image_prompt(self, scene_text: str, visual_analysis: dict = None):
        """Generates a detailed image generation prompt for the scene, prioritizing character consistency."""
        
        # 1. First, identify detailed character descriptions from the vector store
        # We query specifically for physical traits to ensure the "Director" has the right casting info.
        # Reduced k=1 to prevent CUDA OOM on local GPUs with complex prompts.
        character_query = f"Physical appearance, face, hair, and clothing of characters in: {scene_text[:200]}"
        character_context = self.get_context(character_query, k=1)
        
        director_notes = json.dumps(visual_analysis, indent=2) if visual_analysis else "Standard cinematic coverage."

        # 2. Now generate the cinematic prompt
        prompt = PromptTemplate(
            template="""You are the Visionary Director and Cinematographer of this movie. 
            You must describe the SINGLE most iconic shot of this scene for a concept artist.
            
            Key Directive: The faces and characters MUST be consistent with their descriptions.
            
            Casting / Character Bible (Use these details explicitly):
            {context}
            
            Director's Technical Plan (Camera, Mood, Blocking):
            {director_notes}
            
            Scene Action:
            {scene}
            
            Create a highly detailed, photorealistic image generation prompt.
            Structure it as a comma-separated list of visual descriptors.
            Incorporate the Camera Language (angles, lens) and Lighting from the Director's Notes into the technical specs.
            
            MUST Include:
            - SUBJECT: Exact physical details (e.g., "Mark, 19yo male, curly brown hair, pale skin, hoodie, intense introverted expression").
            - ACTION: Subtle acting beat (e.g., "staring intensely at a beer bottle").
            - CAMERA: "Shot on Arri Alexa 65, 35mm lens, f/1.8, cinematic depth of field, bokeh".
            - LIGHTING: "Moody bar lighting, chiaroscuro, rim light on hair, soft shadows".
            - STYLE: "Award-winning movie still, hyperrealistic, 8k, detailed skin texture, film grain".
            
            Output strictly a valid JSON object:
            {{
                "image_prompt": "your detailed prompt string here"
            }}
            """,
            input_variables=["context", "scene", "director_notes"]
        )
        
        chain = prompt | self.llm | JsonOutputParser()
        return chain.invoke({"context": character_context, "scene": scene_text, "director_notes": director_notes})

    def generate_sequence_prompts(self, scene_text: str, visual_analysis: dict):
        """Generates a batch of Storyboard Sketches based on the Shot Plan."""
        
        shot_list = visual_analysis.get("camera_language", {}).get("shot_plan", [])
        if not shot_list:
            # Fallback if no shots defined
            shot_list = [{"shot": "Wide Establish", "reason": "General coverage"}]

        # Get character context once
        character_query = f"Physical appearance of characters in: {scene_text[:200]}"
        character_context = self.get_context(character_query, k=1)
        
        prompt = PromptTemplate(
            template="""You are a professional Storyboard Artist.
            Your task is to draw a sequence of rough pencil sketches for the shots listed below.
            
            Style Reference: 
            - Black and white rough pencil sketch.
            - Loose, dynamic lines.
            - Hand-drawn storyboard aesthetic.
            - High contrast shading.
            
            Characters (Draw them consistent with this):
            {context}
            
            Shot List to Sketch:
            {shots}
            
            Scene Action involved:
            {scene}
            
            For EACH shot in the Shot List, write a specific image generation prompt describing exactly what is drawn in that panel.
            Start every prompt with "Black and white rough pencil sketch of..."
            
            Output strictly a JSON ARRAY of objects:
            [
                {{ "shot_id": 1, "type": "Wide", "image_prompt": "Black and white rough pencil sketch of..." }},
                {{ "shot_id": 2, "type": "Close-up", "image_prompt": "Black and white rough pencil sketch of..." }}
            ]
            """,
            input_variables=["context", "shots", "scene"]
        )
        
        chain = prompt | self.llm | JsonOutputParser()
        try:
            return chain.invoke({
                "context": character_context, 
                "shots": json.dumps(shot_list), 
                "scene": scene_text
            })
        except Exception as e:
            print(f"Sequence Generation failed: {e}")
            return []
