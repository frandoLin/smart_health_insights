from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np

# fixed_size_chunking function splits a document into fixed-size chunks
def fixed_size_chunking(document, chunk_size=256):
    # A naive example: split document by sentences (you might want a more sophisticated splitter)
    sentences = document.split('. ')
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        # Add sentence and a period
        if len(current_chunk.split()) + len(sentence.split()) < chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


# recursive_chunking function splits a document into chunks recursively
def recursive_chunking(document, chunk_size=256):

    len_doc = len(document.split())

    if len_doc <= chunk_size:
        return [document]
    
    mid = len_doc // 2
    left_chunk = document[:mid]
    right_chunk = document[mid:]

    if len(left_chunk.split()) <=chunk_size or len(right_chunk.split()) <= chunk_size:
        return [left_chunk, right_chunk]
    
    return recursive_chunking(left_chunk, chunk_size) + recursive_chunking(right_chunk, chunk_size)


# document-based chunking function splits a document into chunks based on content of the document such as headings, paragraphs, etc.
def document_based_chunking(document, chunk_size=256):
    # A naive example: split document by sentences (you might want a more sophisticated splitter)
    sentences = document.split('. ')
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        # Add sentence and a period
        if len(current_chunk.split()) + len(sentence.split()) < chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


# overlapping_chunking function splits a document into overlapping chunks, which can be useful for tasks like summarization because it allows the model to see more context
def overlapping_chunking(document, chunk_size=256, overlap=50):
    # A naive example: split document by sentences (you might want a more sophisticated splitter)
    sentences = document.split('. ')
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        # Add sentence and a period
        if len(current_chunk.split()) + len(sentence.split()) < chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


# token_based_chunking function splits a document into chunks based on the number of tokens
def token_based_chunking(document, chunk_size=256):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer.tokenize(document)
    chunks = []
    current_chunk = ""
    for token in tokens:
        # Add token and a space
        if len(current_chunk.split()) + len(token.split()) < chunk_size:
            current_chunk += token + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = token + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


# sliding window chunking function splits a document into chunks by sliding a window over the document
def sliding_window_chunking(document, chunk_size=256, stride=50):
    # A naive example: split document by sentences (you might want a more sophisticated splitter)
    sentences = document.split('. ')
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        # Add sentence and a period
        if len(current_chunk.split()) + len(sentence.split()) < chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


# semantic_chunking function splits a document into chunks based on semantic similarity
def semantic_chunking(document, chunk_size=256):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(document, convert_to_tensor=True)
    chunks = []
    current_chunk = ""
    for emb in embeddings:
        # Add embedding
        if len(current_chunk) + len(emb) < chunk_size:
            current_chunk += emb
        else:
            chunks.append(current_chunk)
            current_chunk = emb
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


# dynamic_chunking function splits a document into chunks based on the number of tokens and semantic similarity
def dynamic_chunking(document, chunk_size=256):
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")      
    model = SentenceTransformer('all-MiniLM-L6-v2')
    tokens = tokenizer.tokenize(document)
    chunks = []
    current_chunk = ""
    for token in tokens:
        # Add token and a space
        if len(current_chunk.split()) + len(token.split()) < chunk_size:
            current_chunk += token + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = token + " "
    if current_chunk:   
        chunks.append(current_chunk.strip())        
    embeddings = model.encode(chunks, convert_to_tensor=True)
    new_chunks = []
    current_chunk = ""
    for emb in embeddings:
        # Add embedding
        if len(current_chunk) + len(emb) < chunk_size:
            current_chunk += emb
        else:
            new_chunks.append(current_chunk)
            current_chunk = emb
    if current_chunk:
        new_chunks.append(current_chunk)

    return new_chunks