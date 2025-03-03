import streamlit as st
from langchain_ollama import ChatOllama
from utils import create_inference_chain, retrieve_relevant_docs

# Page configuration
st.set_page_config(
    page_title="Smart Health Insights",
    page_icon="🏥",
    layout="centered"
)

# Title and description
st.title("Smart Health Insights")
st.markdown("Medical information powered by RAG")

# Initialize session state to store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize models (do this once)
@st.cache_resource
def load_models():
    model_embedding = "all-MiniLM-L6-v2"
    model = ChatOllama(model="llama3.1:8b")
    chain = create_inference_chain(model)
    return model_embedding, model, chain

model_embedding, model, chain = load_models()

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Query input
query = st.chat_input("Ask a medical question...")

# Sidebar options
with st.sidebar:
    st.header("Settings")
    show_context = st.checkbox("Show source context", value=False)
    top_k = st.slider("Number of documents to retrieve", min_value=1, max_value=10, value=5)
    use_reranker = st.checkbox("Use reranking", value=True)
    
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# Handle the query
if query:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    # Display assistant response with a spinner
    with st.chat_message("assistant"):
        with st.spinner("Searching medical literature..."):
            # Retrieve context
            rerank_candidates = 20 if use_reranker else -1
            context = retrieve_relevant_docs(
                query=query,
                top_k=top_k,
                rerank_candidates=rerank_candidates,
                model=model_embedding
            )
            
            # Generate response
            message_placeholder = st.empty()
            response = chain.invoke(
                {"prompt": query, "context": context},
                config={"configurable": {"session_id": "unused"}},
            ).strip()
            
            # Display the response
            message_placeholder.markdown(response)
            
            # Show context if requested
            if show_context and context:
                st.write("---")
                st.subheader("Sources")
                for i, ctx in enumerate(context):
                    with st.expander(f"Source {i+1}"):
                        st.markdown(ctx)
    
    # Store assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})