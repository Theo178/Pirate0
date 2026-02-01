const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface ContextAnalysis {
  characters: string[];
  genre: string;
  visual_style: string;
}

export interface UploadResponse {
  message: string;
  chunks_count: number;
  context_analysis: ContextAnalysis;
}

export interface SceneIntent {
  emotion: string;
  story_purpose: string;
}

export interface VisualMood {
  lighting_style: string;
  color_palette: string;
}

export interface ShotPlan {
  shot: string;
  reason: string;
}

export interface CameraLanguage {
  shot_plan: ShotPlan[];
  camera_motion: string;
}

export interface ActorBlocking {
  character: string;
  position: string;
  posture: string;
  eye_focus: string;
  movement: string;
}

export interface SceneAnalysis {
  scene_intent: SceneIntent;
  visual_mood: VisualMood;
  camera_language: CameraLanguage;
  actor_blocking: ActorBlocking[];
  editing_rhythm: Record<string, string>;
  production_logistics: Record<string, string>;
  generated_shots?: SequenceShot[];
}

export interface CharacterArc {
  character: string;
  line: string;
  emotion: string;
  intensity: number;
  sentiment: number;
}

export interface StoryboardImage {
  image_prompt: string;
  image_url: string;
}

export interface SequenceShot {
  shot_id: number;
  type: string;
  prompt: string;
  url: string;
}

export async function uploadScript(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/api/upload-script`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) throw new Error("Failed to upload script");
  return res.json();
}

export async function analyzeScene(sceneText: string): Promise<SceneAnalysis> {
  const res = await fetch(`${API_BASE}/api/analyze-scene`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scene_text: sceneText }),
  });

  if (!res.ok) throw new Error("Failed to analyze scene");
  return res.json();
}

export async function analyzeCharacterArc(sceneText: string): Promise<CharacterArc[]> {
  const res = await fetch(`${API_BASE}/api/analyze-character-arc`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scene_text: sceneText }),
  });

  if (!res.ok) throw new Error("Failed to analyze character arc");
  return res.json();
}

export async function generateStoryboard(sceneText: string): Promise<StoryboardImage> {
  const res = await fetch(`${API_BASE}/api/generate-storyboard`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scene_text: sceneText }),
  });

  if (!res.ok) throw new Error("Failed to generate storyboard");
  return res.json();
}

export async function generateSequence(sceneText: string): Promise<SequenceShot[]> {
  const res = await fetch(`${API_BASE}/api/generate-sequence`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scene_text: sceneText }),
  });

  if (!res.ok) throw new Error("Failed to generate sequence");
  return res.json();
}
