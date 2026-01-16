"""
RAG System - Step 2: Embedding

This script loads chunks from chunks.json, embeds them using 
sentence-transformers, and stores them in a ChromaDB vector database.
"""

import json
from pathlib import Path
import sys
sys.path.insert(0, "/goinfre/abelahse/RAG/libs")

# Embedding and vector store
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


def load_chunks(chunks_file: str | Path) -> list[dict]:
    """Load chunks from JSON file."""
    with open(chunks_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_embeddings_and_store(
    chunks: list[dict],
    collection_name: str = "vllm_docs",
    persist_directory: str = "./chroma_db",
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 100,
):
    """
    Embed all chunks and store them in ChromaDB.
    
    Args:
        chunks: List of chunk dictionaries with 'content' and 'metadata'
        collection_name: Name for the ChromaDB collection
        persist_directory: Where to save the vector database
        model_name: Sentence transformer model to use
        batch_size: Number of chunks to embed at once
    
    Returns:
        The ChromaDB collection
    """
    
    # =========================================================================
    # 1. Load the embedding model
    # =========================================================================
    print(f"🤖 Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"   Embedding dimension: {embedding_dim}")
    
    # =========================================================================
    # 2. Initialize ChromaDB
    # =========================================================================
    print(f"\n📦 Initializing ChromaDB at: {persist_directory}")
    
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Delete existing collection if it exists (fresh start)
    try:
        client.delete_collection(collection_name)
        print(f"   Deleted existing collection: {collection_name}")
    except Exception:
        pass  # Collection doesn't exist
    
    # Create new collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )
    print(f"   Created collection: {collection_name}")
    
    # =========================================================================
    # 3. Embed and store chunks in batches
    # =========================================================================
    print(f"\n🔄 Embedding {len(chunks)} chunks...")
    
    total_chunks = len(chunks)
    
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        batch_end = min(i + batch_size, total_chunks)
        
        # Extract texts and metadata
        texts = [chunk["content"] for chunk in batch]
        metadatas = [chunk["metadata"] for chunk in batch]
        ids = [f"chunk_{j}" for j in range(i, batch_end)]
        
        # Generate embeddings
        embeddings = model.encode(texts, show_progress_bar=False)
        
        # Add to ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
        )
        
        print(f"   Processed {batch_end}/{total_chunks} chunks ({(batch_end/total_chunks)*100:.1f}%)")
    
    print(f"\n✅ Successfully embedded and stored {total_chunks} chunks!")
    
    return collection


def test_search(collection, model_name: str = "all-MiniLM-L6-v2"):
    """Test the vector search with a sample query."""
    
    print("\n" + "=" * 60)
    print("🔍 Testing Search")
    print("=" * 60)
    
    # Load the same model for query embedding
    model = SentenceTransformer(model_name)
    
    # Test queries
    test_queries = [
        "How do I install vLLM?",
        "What is tensor parallelism?",
        "How to configure memory usage?",
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: \"{query}\"")
        print("-" * 40)
        
        # Embed the query
        query_embedding = model.encode([query])[0].tolist()
        
        # Search in ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
        
        # Display results
        for j, (doc, metadata, distance) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            similarity = 1 - distance  # Convert distance to similarity
            print(f"\n  Result {j + 1} (similarity: {similarity:.3f})")
            print(f"  Source: {metadata.get('source', 'unknown')}")
            preview = doc[:150].replace('\n', ' ')
            print(f"  Preview: {preview}...")


if __name__ == "__main__":
    # Paths
    CHUNKS_FILE = Path(__file__).parent / "chunks.json"
    PERSIST_DIR = Path(__file__).parent / "chroma_db"
    MODEL_NAME = "all-MiniLM-L6-v2"  # Fast and good quality
    
    print("=" * 60)
    print("RAG System - Embedding with Sentence Transformers + ChromaDB")
    print("=" * 60)
    
    # Check if chunks file exists
    if not CHUNKS_FILE.exists():
        print(f"❌ Error: {CHUNKS_FILE} not found!")
        print("   Run chunking.py first to generate chunks.")
        exit(1)
    
    # Load chunks
    print(f"\n📄 Loading chunks from: {CHUNKS_FILE}")
    chunks = load_chunks(CHUNKS_FILE)
    print(f"   Loaded {len(chunks)} chunks")
    
    # Embed and store
    collection = create_embeddings_and_store(
        chunks=chunks,
        collection_name="vllm_docs",
        persist_directory=str(PERSIST_DIR),
        model_name=MODEL_NAME,
        batch_size=100,
    )
    
    # Test with sample searches
    test_search(collection, MODEL_NAME)
    
    print(f"\n💾 Vector database saved to: {PERSIST_DIR}")
    print("✨ Embedding complete! Ready for retrieval.")
