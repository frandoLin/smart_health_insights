import os
import faiss
import pickle
from typing import Tuple, List
from data_ingestion.index_doc import index_documents
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
# nltk.download('punkt')

def load_index_and_docs(index_file: str, doc_file: str) -> Tuple[faiss.Index, List[str]]:
    # Load the FAISS index and document list
    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
    else:
        raise ValueError("FAISS index file not found.")
    
    if os.path.exists(doc_file):
        with open(doc_file, "rb") as f:
            documents = pickle.load(f)
    else:
        raise ValueError("Documents file not found.")
    
    return index, documents


def retrieve_relevant_docs(query: str, 
                           top_k: int = 5, 
                           rerank_candidates: int = -1,
                           model: str = "all-MiniLM-L6-v2") -> List[str]:
    
    # Encode the query to get its embedding
    model = SentenceTransformer(model)
    query_embedding = model.encode([query], convert_to_numpy=True)
    index_file_path = "data_ingestion/faiss_index.idx"
    doc_file_path = "data_ingestion/documents.pkl"
    data_path = "data_ingestion/data"
    # Embed the documents and create the FAISS index
    if not os.path.exists(index_file_path) or not os.path.exists(doc_file_path):
        index_documents(data_dir=data_path, index_file=index_file_path, doc_file=doc_file_path)

    # Load the FAISS index and document list
    index, all_documents = load_index_and_docs(index_file_path, doc_file_path)
    # Search for the top k most similar documents to the query
    distances, vector_indices = index.search(query_embedding, rerank_candidates//2 if rerank_candidates != -1 else top_k)

    if len(vector_indices) == 0:
        return []

    if rerank_candidates == -1:
        return [all_documents[idx] for idx in vector_indices[0][:top_k]]

    # Get BM25 candidates
    # nltk.download('punkt', quiet=True)
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in all_documents]
    bm25 = BM25Okapi(tokenized_corpus)
    
    tokenized_query = word_tokenize(query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:rerank_candidates//2]

    # Combine the indices from FAISS and BM25
    combined_indices = list(vector_indices[0]) + bm25_indices
    unique_indices = []
    seen = set()
    for idx in combined_indices:
        if idx not in seen and idx < len(all_documents):
            seen.add(idx)
            unique_indices.append(idx)

    candidates = [all_documents[idx] for idx in unique_indices[:rerank_candidates]] 
    
    # Apply reranker
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [(query, doc) for doc in candidates]
    scores = reranker.predict(pairs)
    
    # Return top-k reranked results
    doc_score_pairs = list(zip(candidates, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    return [doc for doc, _ in doc_score_pairs[:top_k]]


def create_inference_chain(model: str) -> RunnableWithMessageHistory:
    SYSTEM_PROMPT = """
    You are a medical research assistant specializing in synthesizing information from medical literature.
    
    When answering questions:
    1. Base your answers PRIMARILY on the context provided, not your prior knowledge
    2. Use direct quotes or paraphrasing from the context when possible
    3. Cite specific parts from the context to support your statements
    4. If the context has conflicting information, acknowledge this
    5. When the context is insufficient, clearly state what information is missing
    
    Structure your answers in a formal academic style with clear organization.
    """

    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "human",
                "Context: {context}\n\nQuestion: {prompt}"  # Include context in the prompt
            ),
        ]
    )

    chain = prompt_template | model | StrOutputParser()

    chat_message_history = ChatMessageHistory()
    return RunnableWithMessageHistory(
        chain,
        lambda _: chat_message_history,
        input_messages_key="prompt",
        history_messages_key="chat_history",
    )


if __name__ == "__main__":
    context = retrieve_relevant_docs("Show me some research abstracts on liver function abnormalities.", top_k=5)
    print(context)
  
