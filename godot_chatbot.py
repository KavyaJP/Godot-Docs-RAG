import os
import chromadb
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

print("Starting Godot Chatbot setup...")

# --- 1. Configure Models ---
# Set up the LLM and Embedding models to use from Ollama
# Ensure Ollama is running in the background
Settings.llm = Ollama(model="mistral", request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# --- 2. Set Up the Vector Database (ChromaDB) ---
# This is where the knowledge from your documents will be stored persistently.
DB_PATH = "./chroma_db"
if not os.path.exists(DB_PATH):
    print(f"Creating ChromaDB at {DB_PATH}")
    db = chromadb.PersistentClient(path=DB_PATH)
    chroma_collection = db.get_or_create_collection("godot_docs")
else:
    print(f"Loading existing ChromaDB from {DB_PATH}")
    db = chromadb.PersistentClient(path=DB_PATH)
    chroma_collection = db.get_collection("godot_docs")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# --- 3. Index Your Documents (if necessary) ---
# This process converts your documents into vectors and stores them.
# It checks if the database is empty before running to save time on subsequent runs.
if chroma_collection.count() == 0:
    print("No existing documents found in the database. Indexing...")
    
    # Load your documents from the 'godot_docs' directory
    documents = SimpleDirectoryReader("./godot_docs").load_data()
    
    # Create the index from the documents
    # This will take a while the first time!
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
    print(f"Finished indexing. {chroma_collection.count()} chunks indexed.")
else:
    print("Found existing index. Loading from database.")
    # Load the index from the existing storage
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )

# --- 4. Create the Query Engine ---
# This is the main interface for asking questions.
# streaming=True allows the model to output text as it's generated, like ChatGPT.
print("Creating query engine...")
query_engine = index.as_query_engine(streaming=True, similarity_top_k=4)

print("\n--- Godot Documentation Chatbot is Ready! ---")
print("Type your question and press Enter. Type 'exit' to quit.")

# --- 5. Start the Interactive Chat Loop ---
while True:
    prompt = input("\nAsk a Godot question: ")
    if prompt.lower() == "exit":
        print("Exiting chatbot. Goodbye!")
        break

    response = query_engine.query(prompt)
    
    # Print the streaming response
    print("\nGodotBot says:")
    response.print_response_stream()
    print("\n" + "-"*50)