# RAG System for vLLM Documentation

A Retrieval-Augmented Generation (RAG) system that uses the vLLM repository as its knowledge base.

## 🎯 What is RAG?

RAG is a technique that enhances LLM responses by:
1. **Chunking**: Splitting documents into smaller pieces
2. **Embedding**: Converting chunks into numerical vectors
3. **Storing**: Saving vectors in a searchable database
4. **Retrieving**: Finding relevant chunks for a query
5. **Generating**: Using retrieved context to answer questions

## 📁 Project Structure

```
rag_system/
├── README.md           # This file
├── chunking.py         # Step 1: File reading and chunking
├── embedding.py        # Step 2: Creating embeddings (coming soon)
├── vectorstore.py      # Step 3: Vector database (coming soon)
├── retrieval.py        # Step 4: Query and retrieval (coming soon)
├── rag_pipeline.py     # Step 5: Full RAG pipeline (coming soon)
└── chunks.json         # Output: Chunked documents
```

## 🚀 Getting Started

### Step 1: Chunking (Current)

Run the chunking script to process the vLLM repository:

```bash
cd /goinfre/abelahse/RAG
python rag_system/chunking.py
```

This will:
- Read all supported files (.py, .md, .yaml, etc.)
- Split them into chunks using appropriate strategies
- Save the chunks to `chunks.json`

### Chunking Strategies

| Strategy | Best For | Description |
|----------|----------|-------------|
| `FixedSizeChunker` | Simple text | Splits by character count |
| `RecursiveCharacterChunker` | General text | Splits by paragraphs, sentences, words |
| `CodeAwareChunker` | Python code | Splits by classes and functions |
| `MarkdownChunker` | Documentation | Splits by headers |

### Key Parameters

- **chunk_size**: Maximum characters per chunk (default: 1000)
- **chunk_overlap**: Characters shared between chunks (default: 200)

Overlap helps maintain context across chunk boundaries!

## 📊 Understanding the Output

Each chunk contains:
- `content`: The actual text
- `metadata`: Information about the source
  - `source`: Relative file path
  - `file_type`: File extension
  - `chunk_index`: Position in the original file
  - `chunking_strategy`: Which strategy was used

## 🔜 Next Steps

After chunking, we'll implement:

1. **Embeddings** - Convert text to vectors using a model
2. **Vector Store** - Store and search vectors efficiently  
3. **Retrieval** - Find relevant chunks for queries
4. **Generation** - Use an LLM to answer based on context

## 📚 Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Sentence Transformers](https://www.sbert.net/)
