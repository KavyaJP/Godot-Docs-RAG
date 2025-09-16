# Godot Docs RAG Chatbot

This project creates a local, private, and GPU-accelerated chatbot capable of answering questions about the Godot game engine. It uses a Retrieval-Augmented Generation (RAG) pipeline to provide accurate, context-aware answers based on a local collection of Godot documentation.

The chatbot runs entirely on your machine, ensuring your data and queries remain private.

## How It Works

The project leverages a RAG architecture:

1.  **Indexing:** All local documentation files (`.html`, `.md`, etc.) are loaded, split into manageable chunks, and converted into numerical representations (embeddings) using a sentence-transformer model. These embeddings are stored in a local vector database (ChromaDB). This is a one-time process.
2.  **Retrieval:** When a user asks a question, it's also converted into an embedding. The vector database finds the most relevant chunks from the documentation based on semantic similarity.
3.  **Generation:** The retrieved chunks of text are passed as context to a powerful Large Language Model (LLM), which then generates a natural language answer based on the provided information.

## Features

* **Local & Private:** All models and data are stored and processed on your local machine. Nothing is sent to the cloud.
* **GPU Accelerated:** Uses [Ollama](https://ollama.com/) to leverage your local NVIDIA GPU for fast model inference, both for generating embeddings and answers.
* **Easy to Update:** Simply add new documentation files to the data folder and re-run the script to update the knowledge base.
* **Powered by Open-Source:** Built with powerful open-source tools like LlamaIndex, Ollama, and Mistral.

## Technology Stack

* **RAG Framework:** [LlamaIndex](https://www.llamaindex.ai/)
* **LLM & Embedding Models:** [Ollama](https://ollama.com/)
    * **LLM:** `mistral` (or any other model supported by Ollama)
    * **Embedding Model:** `nomic-embed-text`
* **Vector Database:** [ChromaDB](https://www.trychroma.com/)
* **Language:** Python 3.9+

---

## Prerequisites

* Python 3.9 or higher.
* An NVIDIA GPU with CUDA drivers compatible with WSL 2.
* **For Windows users:** WSL 2 must be installed and configured. This project should be run from within a WSL 2 environment (e.g., Ubuntu).

## Setup & Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/KavyaJP/Godot-Docs-RAG.git
    cd Godot-Docs-RAG
    ```

2.  **Install Ollama & Models**
    * Install [Ollama](https://ollama.com/) for your OS. If using WSL, install it *inside* your WSL distribution using the Linux command:
        ```bash
        curl -fsSL [https://ollama.com/install.sh](https://ollama.com/install.sh) | sh
        ```
    * Pull the necessary models. Make sure the Ollama application/server is running.
        ```bash
        ollama pull mistral
        ollama pull nomic-embed-text
        ```

3.  **Set up the Python Environment**
    * Create and activate a virtual environment:
        ```bash
        python -m venv .venv
        source .venv/bin/activate
        ```
    * Install the required Python packages:
        ```bash
        pip install llama-index beautifulsoup4 llama-index-llms-ollama llama-index-embeddings-ollama llama-index-vector-stores-chroma
        ```

4.  **Add Your Documentation**
    * Create a directory named `godot_docs` in the root of the project.
    * Place all your Godot documentation files (HTML, Markdown, text files, etc.) inside this directory.

---

## Running the Chatbot

1.  Ensure the Ollama application or server is running in the background.
2.  Activate your virtual environment (`source .venv/bin/activate`).
3.  Run the main script:
    ```bash
    python godot_chatbot.py
    ```

**Important Note:** The **first time** you run the script, it will begin the indexing process. This can take a significant amount of time depending on the size of your documentation and your hardware. Subsequent runs will be much faster as they will load the existing database.

## How to Use

Once the script initializes and prints `--- Godot Documentation Chatbot is Ready! ---`, you can type your questions into the terminal and press Enter.

To quit the chatbot, type `exit` and press Enter.

## Customization

You can easily customize the models used by editing `godot_chatbot.py`:

* **Change the LLM:** In the `Settings.llm` line, change `model="mistral"` to any other model you have pulled with Ollama (e.g., `llama3`, `phi3`).
* **Change the Embedding Model:** In the `Settings.embed_model` line, change `model_name="nomic-embed-text"` to another embedding model.
* **Adjust Context:** Modify the `similarity_top_k` parameter in `index.as_query_engine(similarity_top_k=4)` to retrieve more or fewer document chunks for context.