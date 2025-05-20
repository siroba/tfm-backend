export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

uvicorn main:app --host 0.0.0.0 --port 8000
