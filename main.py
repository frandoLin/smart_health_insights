from langchain_ollama import ChatOllama
from utils import create_inference_chain, retrieve_relevant_docs


def main():

    model_embedding = "all-MiniLM-L6-v2"
    model = ChatOllama(model="llama3.1:8b")
    chain = create_inference_chain(model)
    
    prompt = "Show me some research abstracts on liver function abnormalities."
    context = retrieve_relevant_docs(query=prompt, top_k=5, rerank_candidates=-1, model=model_embedding)
    # print("Context:\n\n", context)

    response = chain.invoke(
        {"prompt": prompt, "context": context},
        config={"configurable": {"session_id": "unused"}},
    ).strip()
    
    print("Response:\n\n", response)


if __name__ == "__main__":
    main()
  


