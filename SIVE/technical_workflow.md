# Technical Workflow & Architecture

This document outlines the technical implementation of the **Multimodal Scene Intent & Visual Planning Engine**.

## 1. System Architecture

The system follows a modern **Microservices-style** architecture (monolithic deployment, modular design) using **FastAPI** for the backend and **Next.js** for the frontend, connected by a REST API.

```mermaid
graph TD
    Client[Frontend (Next.js)] <-->|REST API| API[FastAPI Backend]
    
    subgraph "Backend Core"
        API --> ScriptProc[Script Processor]
        API --> LLM[LLM Engine (LangChain)]
        
        ScriptProc -->|Chunks| VectorDB[(ChromaDB)]
        LLM <-->|RAG Query| VectorDB
        LLM -->|Inference| Ollama[Ollama (Mistral)]
    end
    
    subgraph "External Services"
        LLM -->|Image Prompts| Poll[Pollinations.AI (Flux Model)]
    end
```

---

## 2. End-to-End Workflow

### Step 1: Script Ingestion & Embedding ( "The Memory" )
**Endpoint:** `POST /api/upload-script`

1.  **Input:** User uploads a PDF, TXT, or MD script file.
2.  **Processing (`script_processor.py`):**
    *   The file is read and text is extracted (using `pypdf` for PDFs).
    *   The text is split into chunks (size: 1000 chars, overlap: 100) using `RecursiveCharacterTextSplitter`.
3.  **Vectorization (`llm_engine.py`):**
    *   Each chunk is converted into a vector embedding using `OllamaEmbeddings` (Mistral model).
    *   These embeddings are stored in a local **ChromaDB** instance.
    *   **Why?** This allows the system to later "remember" details defined early in a script (like a character's hair color) when generating a scene much later.

### Step 2: Scene Analysis ( "The Brain" )
**Endpoint:** `POST /api/analyze-scene`

1.  **Input:** A raw text snippet of a scene.
2.  **RAG Context Retrieval:**
    *   (Optional) The engine queries ChromaDB for relevant context to understand implicit stakes.
3.  **Inference Chain 1: The 6-Layer Director:**
    *   The LLM (Mistral) is prompted with a strict schema to extract 6 layers of filmmaking data:
        1.  **Scene Intent:** (Emotion, Energy)
        2.  **Visual Mood:** (Lighting, Colors)
        3.  **Camera Language:** (Shot types, Lenses)
        4.  **Actor Blocking:** (Positions)
        5.  **Editing Rhythm:** (Pacing)
        6.  **Production Logistics:** (Feasibility)
4.  **Output:** A structured JSON object used by the frontend to display analysis cards.

### Step 3: Visual Generation ( "The Digital Artist" )
**Endpoint:** `POST /api/generate-storyboard` or `/generate-sequence`

1.  **Context Injection (The Secret Sauce):**
    *   Before drawing, the system performs a **RAG Query**: *"Physical appearance of characters in this scene"*.
    *   It retrieves character descriptions (e.g., "Mark, 30s, scar on left cheek") from the vector store.
2.  **Prompt Synthesis:**
    *   The LLM combines:
        *   The **Visual Analysis** (from Step 2 - e.g., "Low key lighting").
        *   The **Character Context** (from RAG).
        *   The **Action** (from the script).
    *   It generates a **Stable Diffusion Prompt** optimized for photorealism (e.g., *"Cinematic wide shot, heavy shadows, Mark with scar looking angry..."*).
3.  **Generation:**
    *   The prompt is encoded and sent to **Pollinations.ai** (using the FLUX model).
    *   The resulting Image URL is returned to the frontend.

---

## 3. Key Technical Decisions

| Component | Choice | Reason |
| :--- | :--- | :--- |
| **Backend Framework** | **FastAPI** | High performance, native IO concurrency for handling concurrent LLM requests. |
| **Orchestrator** | **LangChain** | Manages complex prompt chains and "Output Parsers" to guarantee valid JSON. |
| **LLM Inference** | **Ollama** (Mistral) | Runs locally or on edge, low cost, high privacy. Mistral has excellent reasoning for its size. |
| **Vector Database** | **ChromaDB** | Lightweight, file-based (no server processing needed), perfect for "Project-scoped" memory. |
| **Image Model** | **FLUX (via Pollinations)** | State-of-the-art open-source image model that adheres well to complex prompts. |

## 4. The Innovation: Structured "Implicit" Extraction
Most AI tools maps Input -> Output directly.
*   *Text: "John runs." -> Image: "John running."*

**Our Logic:**
1.  **Input:** "John runs."
2.  **Intermediate Inference (The "Why"):** Is he running *towards* love or *away* from danger?
    *   *AI Decision:* "Away from danger. Fear. High contrast. Shaky cam."
3.  **Final Output:** Image of a panicked John, shaky blur, dark alley.

This intermediate "reasoning layer" is what makes this a **Planning Engine**, not just a generator.
