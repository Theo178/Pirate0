# Multimodal Scene Intent & Visual Planning Engine

> **"Bridging the semantic gap between Script text and Screen visuals."**

## 🎬 Overview
Filmmaking starts with a script, but a script is just text. It lacks the critical visual information—lighting, camera angles, color palettes, and blocking—that turns words into cinema.

This project is an **AI-powered "Digital Cinematographer"**. It doesn't just "read" a script; it *interprets* implicit subtext, extracts emotional intent, and generates rigorous, structured visual planning signals (JSON) to drive storyboards and pre-visualization.

---

## 🚀 Key Features
*   **6-Layer Inference Engine**: Deconstructs scenes into Emotion, Visual Mood, Camera Language, Actor Blocking, Editing Rhythm, and Production Logistics.
*   **Character Consistency (RAG)**: Uses **ChromaDB** to "memorize" character descriptions from the script, ensuring actors look consistent across every generated shot.
*   **Cinematic Prompt Synthesis**: Translates "Director's Notes" into highly technical prompts for the **Flux** image generation model.
*   **Structured Output**: Delivers clean JSON data for integration with 3D engines (Unreal/Unity) or robotic camera systems.

---

## 🛠️ Technology Stack
*   **Core AI**: LangChain (Orchestration), Ollama (Local LLM Inference).
*   **Models**: Mistral 7B (Logic/Reasoning), FLUX.1-schnell (Image Generation).
*   **Memory**: ChromaDB (Vector Store).
*   **Backend**: FastAPI (Python).
*   **Frontend**: Next.js 14, Tailwind CSS, Framer Motion.

---

## 📦 Installation & Setup

### Prerequisites
*   **Python 3.10+**
*   **Node.js 18+**
*   **Ollama** installed and running locally (`ollama pull mistral`)

### 1. Backend Setup
1.  Navigate to the backend directory:
    ```bash
    cd backend
    ```
2.  Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Start the FastAPI server:
    ```bash
    python main.py
    ```
    The API will be live at `http://localhost:8000`.

### 2. Frontend Setup
1.  Navigate to the frontend directory:
    ```bash
    cd frontend
    ```
2.  Install dependencies:
    ```bash
    npm install
    # or
    pnpm install
    ```
3.  Start the development server:
    ```bash
    npm run dev
    ```
    The application will be live at `http://localhost:3000`.

---

## 🎮 Usage Guide
1.  **Upload Script**: Go to the "Dashboard" and upload a PDF or Text file of your screenplay.
2.  **Analyze Scene**: Paste a specific scene snippet into the analyzer.
3.  **View "Director's Mode"**: See the AI break down the scene into Lighting, Camera, and Mood cards.
4.  **Generate Storyboard**: Click "Generate" to visualize the shot using the derived technical/visual data.

---

## 📄 Documentation
For detailed technical info, please check the generated documentation files:
*   [Technical Workflow & Architecture](technical_workflow.md)
*   [Tech Stack Deep Dive](tech_stack_details.md)
*   [Hackathon Presentation Guide](hackathon_presentation.md)
