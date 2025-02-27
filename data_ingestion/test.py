import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize the model with task-specific prompts.
# model = SentenceTransformer(
#     "all-MiniLM-L6-v2",
#     prompts={
#         "classification": "Classify the following text: ",
#         "retrieval": "Retrieve semantically similar text: ",
#     },
#     default_prompt_name="retrieval",
# )

# # Our query text
# query = "What are the benefits of serverless architecture?"

# # Define a list of documents; each document has more than two sentences.
# documents = [
#     (
#         "Serverless computing allows organizations to scale without managing physical infrastructure. "
      
#     ),
#     (
#         "Traditional cloud computing requires managing servers and extensive infrastructure. "
#         "This often results in higher operational costs and complex maintenance routines. "
#         "Organizations must invest significant resources in ensuring system availability."
#     ),
#     (
#         "Modern serverless architectures offer automatic scaling and a pay-as-you-go pricing model. "
#         "They simplify deployment and lower the barrier to entry for new applications. "
#         "Such systems enable rapid development cycles and reduce the overhead of infrastructure management."
#     )
# ]

# # Encode the documents using the default (retrieval) prompt.
# retrieval_embeddings = model.encode(documents)
# print(retrieval_embeddings.shape)
# # Encode the documents using the classification prompt.
# classification_embeddings = model.encode(documents, prompt_name="classification")
# print(classification_embeddings.shape)
# # Compute cosine similarities (for demonstration)
# def compute_similarities(query_emb, doc_embs):
#     sims = np.dot(doc_embs, query_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_embs, axis=1))
#     return sims

# sims_retrieval = compute_similarities(embedding_retrieval, doc_embeddings_retrieval)
# sims_classification = compute_similarities(embedding_classification, doc_embeddings_classification)

# print("Similarity scores using the retrieval prompt:")
# for i, sim in enumerate(sims_retrieval):
#     print(f"Document {i+1}: {sim:.2f}")

# print("\nSimilarity scores using the classification prompt:")
# for i, sim in enumerate(sims_classification):
#     print(f"Document {i+1}: {sim:.2f}")

def split_document(document, max_length=256):
    # A naive example: split document by sentences (you might want a more sophisticated splitter)
    sentences = document.split('. ')
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        # Add sentence and a period
        if len(current_chunk.split()) + len(sentence.split()) < max_length:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

document = ("Serverless computing offers significant benefits. "
            "It reduces operational costs and allows for automatic scaling. "
            "Developers can focus more on writing code instead of managing servers. "
            "This paradigm shift is revolutionizing cloud infrastructure management. "
            "Furthermore, it enables businesses to innovate faster and respond to market changes quickly.")

# Split the document into chunks.
chunks = split_document(document, max_length=15)  # Using word count as a rough proxy for token count
print("Chunks:", chunks)

