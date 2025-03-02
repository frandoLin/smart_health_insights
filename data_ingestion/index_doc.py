# data-ingestion/index_documents.py
# Description: This script is used to generate embeddings for a list of documents using the SentenceTransformer model.
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_ingestion.load_data import load_documents
from data_ingestion.chunk_doc import DocumentChunker
import faiss
from tqdm import tqdm
import pickle

def index_documents(data_dir: str = "./data", index_file: str = "faiss_index.idx", doc_file: str = "documents.pkl") -> None:
    # Load and split documents from the text files.
    documents = load_documents(data_dir)
    if not documents:
        raise ValueError("No documents found. Please check your dataset files.")
    
    print(f"Total documents loaded: {len(documents)}")
    
    chunks = []
    chunker = DocumentChunker()
    print("Chunking documents...")

    for doc in tqdm(documents, desc="Chunking documents", unit="doc"):
        chunks.extend(chunker.chunk(doc, method='overlap', token_limit=True, overlap=50))
    
    # Generate embeddings for each document.
    print("Generating embeddings for documents...")
    embeddings = chunker.model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    
    # Create a FAISS index using the embeddings.
    dim = embeddings.shape[1]
    print("Creating FAISS index...")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    
    # Save the FAISS index and the list of documents.
    print("Saving FAISS index and document metadata...")
    faiss.write_index(index, index_file)
    with open(doc_file, "wb") as f:
        pickle.dump(chunks, f)
    
    print(f"Successfully indexed {len(chunks)} documents!")

if __name__ == "__main__":
    index_documents()
