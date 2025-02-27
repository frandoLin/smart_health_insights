# Smart Health Insights

A Retrieval-Augmented Generation (RAG) system designed to provide accurate health information by retrieving relevant context from trusted medical sources.

## Current Implementation

### Data Processing Pipeline

The project implements a complete data ingestion pipeline for RAG applications:

- **Document Loading**: Loads text documents from files and processes them into manageable units
- **Document Chunking**: Splits documents into semantically meaningful chunks using various strategies
- **Vector Embedding**: Converts text chunks into vector embeddings using the `all-MiniLM-L6-v2` model
- **Vector Database**: Creates a FAISS index for efficient similarity search

### Key Components

- `load_data.py`: Loads and pre-processes documents from text files
- `chunk_doc.py`: Implements the `DocumentChunker` class with multiple chunking strategies
- `index_doc.py`: Creates vector embeddings and builds the FAISS index

### Chunking Strategies

The system supports multiple text chunking approaches:

- **Fixed-size chunking**: Divides text into chunks of approximately equal size
- **Slide window chunking**: Creates chunks with slide window for better context preservation
- **Overlapping chunking**: Creates chunks with overlapping content for better context preservation
- **Semantic chunking**: Groups related content based on semantic similarity
- **Token-based chunking**: Ensures chunks respect token limits of the embedding model (can be combined with the methods above)

## Dependencies
- sentence-transformers
- faiss-cpu (or faiss-gpu)
- tqdm
- numpy
- pickle

## Getting Started

```bash
# Clone the repository
git clone https://github.com/YOUR-USERNAME/smart-health-insights.git
cd smart-health-insights

python data_ingestion/index_doc.py
```

## TODO List
### Short-term Improvements
- **Query Interface**: Implement a query mechanism to search the FAISS index
- **LLM Integration**: Connect with an LLM to generate responses based on retrieved contexts
- **Evaluation**: Add metrics to evaluate retrieval performance and answer quality
- **Add Requirements File**: Create a requirements.txt with all dependencies
### Medium-term Goals
- **Web Interface**: Develop a simple UI for interacting with the system
- **Document Sources**: Add support for more document formats (PDF, HTML, etc.)
- **Enhanced Chunking**: Improve semantic chunking with better similarity measures
- **Metadata Filtering**: Add capability to filter search results by metadata

