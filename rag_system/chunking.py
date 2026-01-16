"""
RAG System - Step 1: File Reading and Chunking (Using LangChain)

This version uses LangChain's built-in loaders and splitters.
Much simpler than doing it from scratch!
"""

import json
from pathlib import Path

# LangChain imports
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language,
)


def load_and_chunk_repository(
    repo_path: str | Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict]:
    """
    Load all files from a repository and chunk them.
    
    Args:
        repo_path: Path to the repository
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of chunks with content and metadata
    """
    repo_path = Path(repo_path)
    all_chunks = []
    
    # =========================================================================
    # 1. Load and chunk Python files (code-aware)
    # =========================================================================
    print("📄 Loading Python files...")
    
    py_loader = DirectoryLoader(
        str(repo_path),
        glob="**/*.py",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
        use_multithreading=True,
    )
    
    try:
        py_docs = py_loader.load()
        print(f"   Found {len(py_docs)} Python files")
        
        # Use Python-aware splitter
        py_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        py_chunks = py_splitter.split_documents(py_docs)
        all_chunks.extend(py_chunks)
        print(f"   Created {len(py_chunks)} chunks from Python files")
    except Exception as e:
        print(f"   Warning: Error loading Python files: {e}")
    
    # =========================================================================
    # 2. Load and chunk Markdown files
    # =========================================================================
    print("📄 Loading Markdown files...")
    
    md_loader = DirectoryLoader(
        str(repo_path),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
        use_multithreading=True,
    )
    
    try:
        md_docs = md_loader.load()
        print(f"   Found {len(md_docs)} Markdown files")
        
        # Use Markdown-aware splitter
        md_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        md_chunks = md_splitter.split_documents(md_docs)
        all_chunks.extend(md_chunks)
        print(f"   Created {len(md_chunks)} chunks from Markdown files")
    except Exception as e:
        print(f"   Warning: Error loading Markdown files: {e}")
    
    # =========================================================================
    # 3. Load and chunk other text files (yaml, txt, rst)
    # =========================================================================
    for ext in ["yaml", "yml", "txt", "rst"]:
        print(f"📄 Loading .{ext} files...")
        
        loader = DirectoryLoader(
            str(repo_path),
            glob=f"**/*.{ext}",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
            use_multithreading=True,
        )
        
        try:
            docs = loader.load()
            if docs:
                print(f"   Found {len(docs)} .{ext} files")
                
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                
                chunks = splitter.split_documents(docs)
                all_chunks.extend(chunks)
                print(f"   Created {len(chunks)} chunks from .{ext} files")
        except Exception as e:
            print(f"   Warning: Error loading .{ext} files: {e}")
    
    print(f"\n✅ Total: {len(all_chunks)} chunks created")
    
    # Convert to simple dict format for saving
    chunks_data = [
        {
            "content": chunk.page_content,
            "metadata": {
                **chunk.metadata,
                "source": str(Path(chunk.metadata.get("source", "")).relative_to(repo_path))
                if chunk.metadata.get("source") else "unknown"
            }
        }
        for chunk in all_chunks
    ]
    
    return chunks_data


if __name__ == "__main__":
    # Path to the vLLM repository
    REPO_PATH = Path(__file__).parent.parent / "vllm"
    OUTPUT_FILE = Path(__file__).parent / "chunks.json"
    
    print("=" * 60)
    print("RAG System - Chunking with LangChain")
    print("=" * 60)
    print(f"Repository: {REPO_PATH}")
    print()
    
    # Process the repository
    chunks = load_and_chunk_repository(
        REPO_PATH,
        chunk_size=1000,
        chunk_overlap=200,
    )
    
    # Show some examples
    print("\n📝 Example chunks:")
    print("-" * 40)
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i + 1}:")
        print(f"  Source: {chunk['metadata'].get('source', 'unknown')}")
        print(f"  Length: {len(chunk['content'])} chars")
        preview = chunk['content'][:150].replace('\n', ' ')
        print(f"  Preview: {preview}...")
    
    # Save to JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Saved {len(chunks)} chunks to: {OUTPUT_FILE}")
