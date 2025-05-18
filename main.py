import os
import uuid
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from pdf2image import convert_from_bytes
from PIL import Image
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from transformers.utils.import_utils import is_flash_attn_2_available
from llama_cpp import Llama
import asyncio

# ==== Config ====
EMBEDDING_MODEL_ID = "vidore/colqwen2-v1.0"
LLM_MODEL_PATH = "models/mistral-7b-instruct-v0.1-q4_k_m.gguf"  # You must download and place this GGUF file

# ==== State ====
sessions = {}

# ==== Init Model ====
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device_str = "cuda" if torch.cuda.is_available() else "cpu"
seconds_per_batch = 5 if device_str == "cuda" else 300  # ~5min por batch en CPU

model = ColQwen2.from_pretrained(
    EMBEDDING_MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map=device,
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()
processor = ColQwen2Processor.from_pretrained(EMBEDDING_MODEL_ID)

llm = Llama(model_path=LLM_MODEL_PATH, n_ctx=2048, n_threads=4, use_mlock=True)

# ==== FastAPI App ====
app = FastAPI()

class Session:
    def __init__(self):
        self.embeddings = []
        self.images = []
        self.messages = []  # chat history


@app.post("/upload_pdf")
async def upload_pdf_stream(file: UploadFile = File(...)):
    pdf_bytes = await file.read()

    async def stream():
        images = convert_from_bytes(pdf_bytes)

        session_id = str(uuid.uuid4())
        session = Session()
        session.images = images

        total_pages = len(images)
        BATCH_SZ = 4
        total_batches = (total_pages + BATCH_SZ - 1) // BATCH_SZ

        estimated_time = total_batches * seconds_per_batch
        yield f"event: eta\ndata: {estimated_time:.1f}\n\n"

        batches = [images[i:i + BATCH_SZ] for i in range(0, total_pages, BATCH_SZ)]
        for i, batch in enumerate(batches):
            yield f"event: progress\ndata: Procesando batch {i + 1}/{total_batches}\n\n"
            batch_inputs = processor.process_images(batch)
            batch_inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}
            with torch.no_grad():
                embs = model(**batch_inputs)
            if isinstance(embs, torch.Tensor):
                for emb in embs:
                    session.embeddings.append(emb.detach().cpu())
            elif isinstance(embs, list):
                for emb in embs:
                    session.embeddings.append(emb.detach().cpu())

        sessions[session_id] = session
        yield f"event: done\ndata: {session_id}\n\n"

    return EventSourceResponse(stream())



@app.post("/chat_stream")
async def chat_stream(request: Request, session_id: str = Form(...), message: str = Form(...)):
    session = sessions.get(session_id)
    if not session:
        return {"error": "Invalid session ID"}

    # Obtener embedding de consulta
    batch_queries = processor.process_queries([message]).to(model.device)
    with torch.no_grad():
        query_embedding = model(**batch_queries)

    # Seleccionar mejor página
    scores = processor.score_multi_vector(query_embedding, session.embeddings)[0]
    best_idx = scores.argmax().item()
    image_info = f"[Retrieved page {best_idx + 1} is used to answer your question.]"

    # Reducir historial para caber en el contexto
    history = session.messages[-4:]  # Limitar a los últimos 4 intercambios
    prompt = "\n".join([f"User: {m['user']}\nAssistant: {m['bot']}" for m in history])
    prompt += f"\nUser: {message}\n{image_info}\nAssistant:"

    print(f"[DEBUG] Prompt length: {len(prompt)} chars")
    print(f"[DEBUG] Prompt preview:\n{prompt[-500:]}")

    # Stream generado
    async def event_generator():
        buffer = ""
        try:
            for token in llm(prompt, stream=True, max_tokens=512):  # Limitar tokens generados
                token_str = token.get("choices", [{}])[0].get("text", "")
                if token_str:
                    buffer += token_str
                    yield {"data": token_str}
                await asyncio.sleep(0.005)
        except Exception as e:
            yield {"event": "error", "data": f"Error: {str(e)}"}

        session.messages.append({"user": message, "bot": buffer.strip()})
        yield {"event": "done", "data": "[DONE]"}

    return EventSourceResponse(event_generator())


