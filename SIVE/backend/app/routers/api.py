from fastapi import APIRouter, UploadFile, HTTPException
from ..services.script_processor import ScriptProcessor
from ..services.llm_engine import LLMEngine
from pydantic import BaseModel

router = APIRouter()
script_processor = ScriptProcessor()
# Lazy initialization or global
llm_engine = LLMEngine() 

class SceneRequest(BaseModel):
    scene_text: str

@router.post("/upload-script")
async def upload_script(file: UploadFile):
    try:
        # Process and chunk
        content = await script_processor.process_file(file)
        chunks = script_processor.chunk_text(content)
        
        # Vectorize
        llm_engine.initialize_vector_store(chunks)
        
        # Analyze high-level context
        context_analysis = llm_engine.analyze_script_context(content)
        
        return {
            "message": "Script processed successfully",
            "chunks_count": len(chunks),
            "context_analysis": context_analysis
        }
    except Exception as e:
        print(f"Error processing script: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import Body, Request

# ... (imports)

@router.post("/analyze-scene")
async def analyze_scene(request: Request):
    try:
        payload = await request.json()
        print(f"Received payload: {payload}")
        scene_text = payload.get("scene_text")
        if not scene_text:
             raise HTTPException(status_code=422, detail="Missing scene_text")
             
        result = llm_engine.analyze_scene(scene_text)
        return result
    except Exception as e:
        print(f"Error analyzing scene: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-character-arc")
async def analyze_character_arc(request: Request):
    try:
        payload = await request.json()
        scene_text = payload.get("scene_text")
        if not scene_text:
             raise HTTPException(status_code=422, detail="Missing scene_text")
             
        result = llm_engine.analyze_character_arc(scene_text)
        return result
    except Exception as e:
        print(f"Error analyzing character arc: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-storyboard")
async def generate_storyboard(request: Request):
    try:
        payload = await request.json()
        scene_text = payload.get("scene_text")
        if not scene_text:
             raise HTTPException(status_code=422, detail="Missing scene_text")
             
        # 1. Generate text prompt
        result = llm_engine.generate_image_prompt(scene_text)
        image_prompt = result.get("image_prompt", "")
        
        if not image_prompt:
            raise HTTPException(status_code=500, detail="Failed to generate image prompt")

        # 2. Construct Pollinations URL
        import urllib.parse
        encoded_prompt = urllib.parse.quote(image_prompt)
        # Using Flux model, 16:9 aspect ratio (approx 1280x720)
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1280&height=720&model=flux&nologo=true"
        
        return {
            "image_prompt": image_prompt,
            "image_url": image_url
        }
    except Exception as e:
        print(f"Error generating storyboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-sequence")
async def generate_sequence(request: Request):
    try:
        payload = await request.json()
        scene_text = payload.get("scene_text")
        if not scene_text:
             raise HTTPException(status_code=422, detail="Missing scene_text")

        # 1. Analyze for shot list
        analysis = llm_engine.analyze_scene(scene_text)
        
        # 2. Generate prompts for sequence
        prompts_list = llm_engine.generate_sequence_prompts(scene_text, analysis)
        
        # 3. Generate URLs
        results = []
        import urllib.parse
        import time
        import random
        
        for item in prompts_list:
            prompt = item.get("image_prompt", "")
            if prompt:
                # Add a 2-second delay to respect Pollinations.ai anonymous rate limits
                time.sleep(2)
                
                encoded = urllib.parse.quote(prompt)
                seed = random.randint(0, 999999)
                # Append seed to bypass cache or collisions, and ensure fresh generation
                url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=576&model=flux&nologo=true&seed={seed}"
                
                results.append({
                    "shot_id": item.get("shot_id"),
                    "type": item.get("type"),
                    "prompt": prompt,
                    "url": url
                })
        
        return results

    except Exception as e:
        print(f"Error generating sequence: {e}")
        raise HTTPException(status_code=500, detail=str(e))
