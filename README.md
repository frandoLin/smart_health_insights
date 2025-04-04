# Smart Health Insights

A Retrieval-Augmented Generation (RAG) system designed to provide accurate health information by retrieving relevant context from trusted medical sources.

## Current Implementation

### Dataset

This project uses the **PubMed 20k RCT** dataset, which includes:
- 20,000 medical research abstracts of randomized controlled trials
- Sourced from the PubMed database of biomedical literature
- Rich content covering various medical conditions, treatments, and clinical outcomes
- Structured abstracts with sections (background, methods, results, conclusions)

The dataset provides high-quality, scientific medical information that serves as the knowledge base for this RAG system.


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
- `utils.py`: Provides core RAG functionality including hybrid retrieval (vector + BM25), reranking, and LLM chain creation

### Chunking Strategies

The system supports multiple text chunking approaches:

- **Fixed-size chunking**: Divides text into chunks of approximately equal size
- **Slide window chunking**: Creates chunks with slide window for better context preservation
- **Overlapping chunking**: Creates chunks with overlapping content for better context preservation
- **Semantic chunking**: Groups related content based on semantic similarity
- **Token-based chunking**: Ensures chunks respect token limits of the embedding model (can be combined with the methods above)


## Dependencies
- RTX 3090 GPU 24G
- Linux ubuntu 20.04
- Python 3.10
- node v18.20.7
- npm 10.8.2
- React 18.3.1
- Vite 5.4.14
- Axios 1.8.1


## Getting Started

```bash
# Clone the repository
git clone https://github.com/frandoLin/smart-health-insights.git
cd smart-health-insights

pip install -r requirements.txt

python main.py
```

## Web Interface Options

The project provides two options for a web-based interface to interact with the RAG system:

### Option 1: Streamlit Interface (Recommended)

A simple, Python-based interface with chat history and advanced query options:

```bash
# Install Streamlit
pip install streamlit

# Run the Streamlit app
streamlit run app.py --server.address 0.0.0.0
```
![Streamlit Interface Screenshot](images/streamlit_interface.png)

### Option 2: React Frontend with FastAPI Backend

A more customizable web interface with separate frontend and backend:

```bash
# Start the FastAPI backend
cd smart-health-insights
python api.py

# In a separate terminal, start the React frontend
cd smart-health-insights/react
npm install  # First time only
npm run dev

```
![React Interface Screenshot](images/react_interface.png)

## TODO List
### Short-term Improvements
- ~~**Query Interface**: Implement a query mechanism to search the FAISS index~~
- ~~**LLM Integration**: Connect with an LLM to generate responses based on retrieved contexts~~
- **Evaluation**: Add metrics to evaluate retrieval performance and answer quality
- ~~**Add Requirements File**: Create a requirements.txt with all dependencies~~
### Medium-term Goals
- ~~**Web Interface**: Develop a simple UI for interacting with the system~~
- **Document Sources**: Add support for more document formats (PDF, HTML, etc.)
- **Enhanced Chunking**: Improve semantic chunking with better similarity measures
- **Metadata Filtering**: Add capability to filter search results by metadata

