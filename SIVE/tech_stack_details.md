# Technology Stack & Engineering Rationale

This document provides a deep technical analysis of the tools, frameworks, and libraries used to build the **Multimodal Scene Intent Engine**. It explains *why* each technology was chosen over alternatives.

---

## 1. The Core AI Engine (The Brain)

### **LangChain**
*   **What it is:** A framework for developing applications powered by language models.
*   **Role:** Acts as the "Operating System" for our backend. It manages the `PromptTemplate` -> `LLM` -> `Parser` data flow (LCEL).
*   **Why we chose it:**
    *   **Abstraction:** It decouples our logic from the specific model. We can switch from Mistral to GPT-4 or Claude with a single line of code.
    *   **RAG Standards:** It provides pre-built interfaces for Vector Stores (Chroma) and Embeddings, saving weeks of boilerplate code.
    *   **Structured Output:** Its `JsonOutputParser` is critical for ensuring the LLM outputs valid JSON that our frontend can render, handling retry logic if the JSON is malformed.

### **Ollama (running Mistral)**
*   **What it is:** A tool to run open-source LLMs locally.
*   **Role:** The primary inference engine for "Scene Analysis" and "Prompt Synthesis".
*   **Why we chose it:**
    *   **Data Privacy:** Process scripts locally without sending sensitive IP to the cloud.
    *   **Cost:** Completely free inference.
    *   **Latency:** No network round-trips for the heavy textual analysis.
    *   **Model Choice (Mistral):** Mistral 7B is the current state-of-the-art for "small" models. It outperforms Llama-2-13b on many reasoning benchmarks, making it perfect for deducing *implicit intent*.

### **ChromaDB**
*   **What it is:** A native, open-source embedding database.
*   **Role:** The "Long-Term Memory" of the application. It stores chunks of the script as vectors.
*   **Why we chose it:**
    *   **Serverless/Embedded:** Unlike Pinecone or Weaviate, Chroma can run directly inside our Python process (specifically `backend/chroma_db`). It requires no Docker containers or external API keys, reducing infrastructure complexity by 100%.
    *   **Simplicity:** Perfect for "Session-based" RAG where each upload creates a temporary, isolated knowledge base.

### **Pollinations.AI (FLUX Model)**
*   **What it is:** A decentralized, free API for image generation.
*   **Role:** The "Visualizer". It takes our enhanced prompts and generates the storyboards.
*   **Why we chose it:**
    *   **FLUX Integration:** It offers access to the **FLUX.1-schnell** model, which is currently superior to Stable Diffusion XL (SDXL) for following complex, text-heavy prompts.
    *   **No Auth/Rate Limits:** For a hackathon/MVP, avoiding credit card sign-ups and API rate limits is crucial. Pollinations allows rapid iteration.

---

## 2. The Backend (The Nervous System)

### **FastAPI**
*   **What it is:** A modern, fast (high-performance) web framework for building APIs with Python 3.8+.
*   **Role:** Exposes the AI logic as REST endpoints (`/analyze-scene`, `/generate-storyboard`).
*   **Why we chose it:**
    *   **Native Async Support:** LLM operations are I/O bound (waiting for model response). FastAPI's `async def` allows the server to handle multiple user requests simultaneously while waiting for the LLM, unlike Flask or Django which block by default.
    *   **Pydantic Integration:** We define our data schemas (e.g., `SceneRequest`) once. FastAPI automatically validates all incoming JSON against these schemas, preventing "AttributeError" crashes deep in the code.
    *   **Swagger UI:** It auto-generates interactive documentation at `/docs`, making it easy for frontend devs to test endpoints without writing curl commands.

### **PyPDF**
*   **What it is:** A pure-python PDF library.
*   **Role:** extracting text from script files.
*   **Why we chose it:**
    *   **Reliability:** It handles the complex layout of film scripts (indentations for dialogue vs. action) better than generic text extractors.

---

## 3. The Frontend (The Face)

### **Next.js 14**
*   **What it is:** The React framework for the web.
*   **Role:** Renders the UI, manages client state, and displays the generated assets.
*   **Why we chose it:**
    *   **Server-Side Rendering (SSR):** Typically scripts are heavy. We can process some logic on the server.
    *   **Routing:** File-based routing (`app/page.tsx`) makes it easy to organize the "Upload", "Dashboard", and "Storyboard" views.

### **Tailwind CSS**
*   **What it is:** A utility-first CSS framework.
*   **Role:** Styling the application.
*   **Why we chose it:**
    *   **Speed:** We can build a "dark mode" cinematic UI in minutes using classes like `bg-zinc-950 text-zinc-100`.
    *   **Consistency:** Ensures margins and colors are uniform across the app.

### **Framer Motion** (Suggested/Potential)
*   **What it is:** A motion library for React.
*   **Role:** Handling the "Loading" animations and the smooth reveal of storyboard cards.
*   **Why:** In an AI app, "waiting" is inevitable. High-quality animations (like a pulsing "Analyzing..." state) make the wait feel like "processing" rather than "lagging".

---

## Summary of the "Secret Sauce" Workflow
1.  **FastAPI** receives the Script.
2.  **PyPDF** extracts text.
3.  **LangChain** chunks it.
4.  **ChromaDB** embeds it (using **Ollama**).
5.  User requests a scene.
6.  **LangChain** retrieves context from **ChromaDB**.
7.  **Ollama (Mistral)** analyzes intent and visualizes the shot.
8.  **Pollinations (FLUX)** renders the final pixel data.
9.  **Next.js** displays the result.
