import os
import uuid
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pdf2image import convert_from_bytes
from PIL import Image
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from transformers.utils.import_utils import is_flash_attn_2_available # Keep for now
from llama_cpp import Llama
import asyncio
import time

# ==== Config ====
EMBEDDING_MODEL_ID = "vidore/colqwen2-v1.0"
LLM_MODEL_PATH = "models/mistral-7b-instruct-v0.1-q4_k_m.gguf"
SECONDS_PER_PAGE_PROCESSING = 150 # Adjust after testing on AMD

# ==== State ====
sessions = {}

# ==== Init Model ====
# For AMD GPUs with ROCm-enabled PyTorch, torch.cuda.is_available() should work.
# The device name will still be 'cuda:0' even for AMD GPUs if ROCm is correctly set up.
if torch.cuda.is_available():
    device = "cuda:0" # PyTorch uses 'cuda' as the device string for ROCm as well
    # RX 6600 (RDNA2) has good FP16 support. BF16 might be less optimal or unsupported.
    torch_dtype = torch.float16
    print(f"PyTorch ROCm device available: {torch.cuda.get_device_name(0)}")
else:
    print(f"No ROCm device found by PyTorch, falling back to CPU.")
    device = "cpu"
    torch_dtype = torch.float32

print(f"Using device: {device} with dtype: {torch_dtype}")

# Determine attention implementation
# For ROCm, 'sdpa' (scaled_dot_product_attention) is generally a good choice if flash_attn_2 isn't available/optimal.
attn_implementation = "sdpa" # Default to scaled_dot_product_attention
if device.startswith("cuda") and is_flash_attn_2_available():
    # This check might still be relevant if a ROCm-compatible flash attention variant exists and is picked up.
    # However, 'sdpa' is often more robust across hardware.
    # If issues, explicitly set to "sdpa" or None for AMD.
    attn_implementation = "flash_attention_2"
    print("Using flash_attention_2")
else:
    print(f"Using attn_implementation: {attn_implementation}")


model = ColQwen2.from_pretrained(
    EMBEDDING_MODEL_ID,
    torch_dtype=torch_dtype,
    device_map=device, # This should correctly map to your AMD GPU if device is "cuda:0" via ROCm
    attn_implementation=attn_implementation,
).eval()
processor = ColQwen2Processor.from_pretrained(EMBEDDING_MODEL_ID)

# Initialize Llama.cpp with GPU offloading for AMD (requires llama-cpp-python compiled with ROCm)
try:
    llm = Llama(
        model_path=LLM_MODEL_PATH,
        n_ctx=2048,
        n_threads=os.cpu_count() or 4, # Use a reasonable number of CPU threads
        n_gpu_layers=-1, # Offload all possible layers to GPU. Adjust if OOM or issues.
                         # Start with e.g. 30-35 for a 8GB VRAM card like RX 6600 with a 7B Q4 model.
                         # Check VRAM usage. -1 tries to offload all.
        use_mlock=True, # Can help keep model in RAM, check if it causes issues
        verbose=False # Set to True for debugging llama.cpp initialization
    )
    print("Llama model loaded with GPU offloading (ROCm).")
except Exception as e:
    print(f"Failed to load Llama model with GPU offloading, falling back to CPU-only: {e}")
    print("Make sure you have llama-cpp-python installed with ROCM/HIPBLAS support.")
    llm = Llama(
        model_path=LLM_MODEL_PATH,
        n_ctx=2048,
        n_threads=os.cpu_count() or 4,
        use_mlock=True,
        verbose=False,
        n_gpu_layers=0 # No layers to GPU
    )
    print("Llama model loaded on CPU.")

# ==== FastAPI App ====
app = FastAPI(title="PDF Chat API")

FRONTEND_DEV_ORIGIN = "http://localhost:5173"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


class Session:
    def __init__(self):
        self.embeddings: List[torch.Tensor] = []
        self.images: List[Image.Image] = []
        self.messages: List[dict] = []


@app.post("/upload_pdf")
async def upload_pdf_stream(file: UploadFile = File(...)):
    pdf_bytes = await file.read()

    async def stream_generator():
        session_id = str(uuid.uuid4())
        session = Session()

        yield f"event: status\ndata: Starting PDF processing...\n\n"

        # 1. PDF to Image Conversion
        yield f"event: status\ndata: Converting PDF to images (this may take a moment for large PDFs).\n\n"
        conversion_start_time = time.time()
        try:
            images = convert_from_bytes(pdf_bytes, timeout=300)  # Added timeout
            session.images = images
        except Exception as e:
            yield f"event: error\ndata: PDF to image conversion failed: {str(e)}\n\n"
            return

        total_pages = len(images)
        if total_pages == 0:
            yield f"event: error\ndata: No pages found in PDF.\n\n"
            return

        conversion_duration = time.time() - conversion_start_time
        yield f"event: status\ndata: PDF converted to {total_pages} image(s) in {conversion_duration:.2f}s. Starting embedding generation.\n\n"

        # 2. Embedding Generation (with page-by-page progress)
        estimated_embedding_time = total_pages * SECONDS_PER_PAGE_PROCESSING
        total_estimated_time = conversion_duration + estimated_embedding_time  # More accurate ETA
        yield f"event: eta\ndata: {total_estimated_time:.1f}\n\n"

        BATCH_SZ = 1  # Adjust batch size based on VRAM and model preference
        page_count_offset = 0

        # Prepare batches of images
        image_batches = [images[i:i + BATCH_SZ] for i in range(0, total_pages, BATCH_SZ)]

        for batch_idx, batch_pil_images in enumerate(image_batches):
            num_images_in_batch = len(batch_pil_images)
            # yield f"event: status\ndata: Processing image batch {batch_idx + 1}/{len(image_batches)}...\n\n"

            try:
                batch_inputs = processor.process_images(batch_pil_images)  # list of PIL images
                batch_inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}
                with torch.no_grad():
                    # Assuming model returns embeddings for all images in the batch
                    batch_embs_output = model(**batch_inputs)

                # Standardize embeddings to a list of CPU tensors
                individual_embs_in_batch = []
                if isinstance(batch_embs_output, torch.Tensor):
                    if batch_embs_output.ndim == 1 and num_images_in_batch == 1:  # Single embedding for a single image batch
                        individual_embs_in_batch.append(batch_embs_output.detach().cpu())
                    elif batch_embs_output.ndim > 1 and batch_embs_output.shape[
                        0] == num_images_in_batch:  # Multiple embeddings in one tensor
                        for i in range(num_images_in_batch):
                            individual_embs_in_batch.append(batch_embs_output[i].detach().cpu())
                    else:
                        raise ValueError(
                            f"Unexpected embedding tensor shape: {batch_embs_output.shape} for batch size {num_images_in_batch}")
                elif isinstance(batch_embs_output, list) and len(batch_embs_output) == num_images_in_batch:
                    for emb_tensor in batch_embs_output:
                        individual_embs_in_batch.append(emb_tensor.detach().cpu())
                else:
                    raise ValueError(
                        f"Unexpected embedding output type ({type(batch_embs_output)}) or count for batch.")

                # Report progress for each page in the processed batch
                for i in range(num_images_in_batch):
                    current_page_absolute_idx = page_count_offset + 1 + i
                    session.embeddings.append(individual_embs_in_batch[i])
                    yield f"event: page_progress\ndata: {current_page_absolute_idx}/{total_pages}\n\n"

                page_count_offset += num_images_in_batch

            except Exception as e:
                yield f"event: error\ndata: Error processing page embeddings (around page {page_count_offset + 1}): {str(e)}\n\n"
                # Decide if you want to stop or continue. For now, stop.
                return

        sessions[session_id] = session
        yield f"event: done\ndata: {session_id}\n\n"

    return EventSourceResponse(stream_generator())


@app.post("/chat_stream")
async def chat_stream(request: Request, session_id: str = Form(...), message: str = Form(...)):
    session = sessions.get(session_id)
    if not session:
        async def error_stream():
            yield {"event": "error", "data": "Invalid session ID"}

        return EventSourceResponse(error_stream())

    if not session.embeddings:
        async def error_stream():
            yield {"event": "error", "data": "No document processed for this session or processing failed."}

        return EventSourceResponse(error_stream())

    try:
        batch_queries = processor.process_queries([message]).to(model.device)
        with torch.no_grad():
            query_embedding = model(**batch_queries)
    except Exception as e:
        async def error_stream():
            yield {"event": "error", "data": f"Error processing query: {str(e)}"}

        return EventSourceResponse(error_stream())

    try:
        if not session.embeddings:  # Should be caught above, but defensive check
            raise ValueError("Session embeddings are empty.")
        all_doc_embeddings = torch.stack(session.embeddings)
        scores = processor.score_multi_vector(query_embedding, all_doc_embeddings)[0]
        best_idx = scores.argmax().item()
        image_info = f"[Retrieved page {best_idx + 1} to answer the question.]"
    except Exception as e:
        image_info = "[Could not retrieve a specific page; answering generally.]"
        print(f"Error during page retrieval/scoring: {e}")

    history = session.messages[-4:]
    prompt_parts = []
    for turn in history:
        prompt_parts.append(f"User: {turn['user']}")
        prompt_parts.append(f"Assistant: {turn['bot']}")
    prompt_parts.append(f"User: {message}")
    prompt_parts.append(image_info)
    prompt_parts.append("Assistant:")
    prompt = "\n".join(prompt_parts)

    # print(f"[DEBUG] Prompt length for LLM: {len(prompt.encode('utf-8'))} bytes, {len(prompt)} chars")
    # print(f"[DEBUG] Prompt preview (last 500 chars):\n{prompt[-500:]}")

    async def event_generator():
        response_buffer = ""
        try:
            stream_output = llm(prompt, stream=True, max_tokens=512, temperature=0.7)
            for chunk in stream_output:
                token_str = chunk.get("choices", [{}])[0].get("text", "")
                if token_str:
                    response_buffer += token_str
                    yield {"data": token_str}
                await asyncio.sleep(0.005)
        except Exception as e:
            print(f"LLM streaming error: {e}")
            yield {"event": "error", "data": f"LLM Error: {str(e)}"}

        session.messages.append({"user": message, "bot": response_buffer.strip()})
        yield {"event": "done", "data": "[END_OF_STREAM]"}

    return EventSourceResponse(event_generator())

# To run: uvicorn main:app --reload --port 8000
# Ensure poppler is installed for pdf2image (e.g., `conda install -c conda-forge poppler` or `sudo apt-get install poppler-utils`)