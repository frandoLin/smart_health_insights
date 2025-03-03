from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from utils import create_inference_chain, retrieve_relevant_docs
import uvicorn

app = FastAPI(title="Smart Health Insights API")

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://192.168.1.158:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request and response models
class QueryRequest(BaseModel):
    query: str
    show_context: bool = False
    top_k: int = 5
    rerank_candidates: int = -1

class QueryResponse(BaseModel):
    answer: str
    context: list[str] = None

# Initialize model and chain once (could move to a startup event if needed)
model_embedding = "all-MiniLM-L6-v2"
model = ChatOllama(model="llama3.1:8b")
chain = create_inference_chain(model)

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        # Retrieve relevant context
        context = retrieve_relevant_docs(
            query=request.query, 
            top_k=request.top_k, 
            rerank_candidates=request.rerank_candidates, 
            model=model_embedding
        )
        
        # Generate response using the chain
        response = chain.invoke(
            {"prompt": request.query, "context": context},
            config={"configurable": {"session_id": "unused"}},
        ).strip()
        
        # Return the response and context if requested
        return QueryResponse(
            answer=response,
            context=context if request.show_context else None
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)