from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request

from pydantic import BaseModel

from app.ingestion import ingest_document
from app.retrieval import retrieve_chunks
from app.llm import generate_answer

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="RAG-Based Question Answering System",
    description="Upload documents and ask questions using RAG",
    version="1.0.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ----------- Request Models -----------

class QueryRequest(BaseModel):
    question: str

# ----------- API Routes -----------

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    ingest_document(file)
    return {"message": "Document ingested successfully"}


@app.post("/query")
@limiter.limit("5/minute")
def ask_question(request: Request, payload: QueryRequest):
    chunks = retrieve_chunks(payload.question)

    if not chunks:
        return {
            "answer": "No relevant information found in the uploaded document."
        }

    answer = generate_answer(chunks, payload.question)
    return {"answer": answer}


