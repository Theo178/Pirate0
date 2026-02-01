# Hackathon Presentation Guide: Multimodal Scene Intent & Visual Planning Engine

**Time Limit:** 10 Minutes using the 10-20-30 Rule (approximate).
**Audience:** Jury Members (Technical + Creative/Product backgrounds).
**Core Innovation:** Bridging the gap between *Textual Script* and *Visual Emotion* using a multi-layered AI inference engine.

---

## Part 1: The Hook & The "Why" (0:00 - 2:00)
**Goal:** Establish the problem in the **Cinema Domain**.

**Slide 1: Title Slide**
*   **Visual:** A split screen. Left side: A boring script page. Right side: A stunning, moody movie shot (e.g., from *Blade Runner* or *The Social Network*).
*   **Spoken:** "Great movies aren't made on the page; they are made in the translation from page to screen. A script says 'He sits in the room.' But a Director of Photography (DoP) asks: 'Is he lonely? Is he powerful? Is he hiding?'"

**Slide 2: The Problem - The Semantic Gap**
*   **Visual:** A diagram showing "Script (Text)" -> [?] -> "Movie (Visuals)".
*   **Spoken:** "Current AI image generators are literal. If you type 'Man sits in chair', you get a man in a chair. But in filmmaking, that's useless. We need **Intent**. We need to know if the lighting should be harsh or soft, if the camera is high or low.
*   **Key Phrase:** "Scripts contain implicit emotional data that standard generative AI simply ignores. We built a system to extract that soul."

---

## Part 2: The Innovation - The "Digital Cinematographer" (2:00 - 4:00)
**Goal:** Explain *what* you built (The Concept).

**Slide 3: The 6-Layer Inference Engine**
*   **Visual:** A diagram showing the Script Text going into a Brain (The Engine) and splitting into 6 clear streams (use the structure found in your `llm_engine.py`):
    1.  **Scene Intent** (Emotion, Energy)
    2.  **Visual Mood** (Lighting, Contrast)
    3.  **Camera Language** (Shot type, lens choice)
    4.  **Actor Blocking** (Position, Body language)
    5.  **Editing Rhythm** (Pacing)
    6.  **Production Logistics** (Feasibility)
*   **Spoken:** "We didn't just ask an LLM to 'draw the scene'. We engineered a pipeline that acts like a film crew. One agent mimics the Director (determining emotion), one mimics the DoP (choosing lenses and lighting), and one acts as the Casting Director (maintaining character consistency)."

**Slide 4: Structured Visual Planning Signals**
*   **Visual:** Show the raw JSON output. Highlight fields like `"emotion": "Suspicious"`, `"lighting_style": "Chiaroscuro"`, `"camera_motion": "Slow push-in"`.
*   **Spoken:** "This is the innovation. We convert vague art into **Structured Data**. This JSON isn't just metadata; it's a blueprint that can drive storyboard generators, 3D engines like Unreal Engine, or robotic camera arms."

---

## Part 3: Technical Domain Deep Dive (4:00 - 7:00)
**Goal:** Satisfy the technical judges. Explain *how* it works.

**Slide 5: Architecture Stack**
*   **Visual:** Architecture Diagram.
    *   **Input:** PDF/Text Script.
    *   **Orchestration:** LangChain + Python Backend.
    *   **The Brain:** Mistral (via Ollama) specialized for creative inference.
    *   **Memory (RAG):** ChromaDB. (Crucial point!).
    *   **Visualization:** Stable Diffusion (or compatible API) primed with "Director's Notes".
*   **Spoken:** "Technically, this is a multi-agent chain. We use **RAG (Retrieval Augmented Generation)** to solve the biggest problem in GenAI video: Consistency. By storing character details (embedding physical descriptions) in ChromaDB, our engine ensures 'Mark' looks like 'Mark' in every generated storyboard panel."

**Slide 6: The "Prompt Engineering" Secret**
*   **Visual:** Show the "Translation" layer.
    *   *Input:* "Mark looks at the beer."
    *   *Derived Intent:* "Regret, Low-Key Lighting, 45-degree shutter."
    *   *Final Prompt:* "Cinematic wide shot, Arri Alexa, low-key lighting, Mark (pale, hoodie) staring at beer..."
*   **Spoken:** "We developed a 'Cinematic Prompt Synthesis' layer. It takes the structured JSON signals and compiles them into highly technical photography prompts, creating 8K photorealistic visualizations that match the *mood*, not just the nouns."

---

## Part 4: Live Demo (7:00 - 9:00)
**Goal:** Prove it works.

**Action:** Switch to the Frontend.
1.  **Step 1:** Paste a raw scene script. (Pick something emotional, not just action).
2.  **Step 2:** Click "Analyze". Show the "Thinking" state.
3.  **Step 3:** **The Reveal.** Show the **JSON data** first ("Look, it understood the scene is 'Melancholic'").
4.  **Step 4:** Show the **Generated Storyboard/Images**.
    *   *Commentary:* "Notice the lighting matches the mood. Notice the camera angle emphasizes the power dynamic we extracted."
5.  **Step 5 (Optional):** Show the "Sequence" feature (Storyboard mode).

---

## Part 5: Conclusion & Future (9:00 - 10:00)
**Goal:** Leave a lasting impression.

**Slide 7: Impact & Future**
*   **Visual:** "From Script to Screen in Seconds."
*   **Bullet Points:**
    *   Rapid Prototyping for Indie Filmmakers.
    *   Automated Shot Lists for Crews.
    *   Integration with Unreal Engine meant for Pre-viz.
*   **Spoken:** "We are effectively democratizing the *Director's Vision*. We aren't replacing artists; we are giving them a visual language to communicate their ideas instantly."

**Closing Statement:**
"This is the Multimodal Scene Intent Engine. It turns text into feelings, and feelings into frames. Thank you."

---

## Key Terms to Stress (Cheat Sheet)

*   **Cinema Domain:** Mise-en-scène, Diegetic elements, Visual Tone, Color Grading, Emotional Resonance, Camera Blocking.
*   **Technical Domain:** Multimodal Inference, Structured Output Parsing (JSON), Retrieval Augmented Generation (RAG), Latent Space, Zero-shot Classification, Vector Embeddings (ChromaDB).
