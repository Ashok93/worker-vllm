# Keep your existing base
FROM nvidia/cuda:12.1.0-base-ubuntu22.04 

# --- ADDITION 1: System Dependencies ---
# PaddleOCR and OpenCV require these libraries to handle images/math
RUN apt-get update -y && apt-get install -y \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN ldconfig /usr/local/cuda-12.1/compat/

# Install Python dependencies (from your updated requirements.txt)
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade -r /requirements.txt

# --- ADDITION 2: vLLM & Specific Paddle Wheel ---
# Note: Using vLLM 0.6.2+ is recommended for the latest VLM support
RUN python3 -m pip install vllm==0.6.2 && \
    python3 -m pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4

# Setup for Option 2: Building the Image with the Model included
ARG MODEL_NAME="PaddlePaddle/PaddleOCR-VL"
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=1 

ENV PYTHONPATH="/:/vllm-workspace"

COPY src /src

# --- ADDITION 3: Pre-download Layout Models ---
# This "bakes" the PP-DocLayoutV2 models into the image
RUN python3 -c "from paddleocr import PPStructure; PPStructure(layout=True, show_log=False)"

RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
    export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
    python3 /src/download_model.py; \
    fi

# Start the handler
CMD ["python3", "/src/handler.py"]