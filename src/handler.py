import os
import cv2
import base64
import requests
import numpy as np
import tempfile
import runpod
from paddleocr import PPStructure
from utils import JobInput
from engine import vLLMEngine, OpenAIvLLMEngine

# Initialize Engines
vllm_engine = vLLMEngine()
openai_engine = OpenAIvLLMEngine(vllm_engine)

# Initialize Layout Engine (DocLayoutV2)
layout_engine = PPStructure(layout=True, table=False, ocr=False, show_log=False)

def get_file_path(input_data):
    """
    Decides if input is a URL, Base64, or local path.
    Saves to a temporary file and returns the path.
    """
    if input_data.startswith('http'):
        # It's a URL
        resp = requests.get(input_data)
        suffix = ".pdf" if "pdf" in input_data.lower() else ".jpg"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(resp.content)
        return temp_file.name
    
    elif len(input_data) > 100: 
        # Assume it's a Base64 string
        # Clean prefix if exists (e.g., 'data:image/png;base64,')
        if "," in input_data:
            input_data = input_data.split(",")[1]
        
        decoded_data = base64.b64decode(input_data)
        # We try to guess if it's PDF or Image based on headers
        suffix = ".pdf" if decoded_data.startswith(b'%PDF') else ".jpg"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(decoded_data)
        return temp_file.name
    
    return input_data # Assume it's already a local path

async def handler(job):
    job_input = JobInput(job["input"])
    use_layout = job["input"].get("use_layout", False)
    raw_data = job["input"].get("image") or job["input"].get("image_url") or job["input"].get("pdf")

    if not use_layout:
        # Standard route: Let vLLM handle it (usually supports image_url)
        engine = openai_engine if job_input.openai_route else vllm_engine
        results_generator = engine.generate(job_input)
        async for batch in results_generator:
            yield batch
        return

    # --- Layout + VLM Workflow ---
    file_path = get_file_path(raw_data)
    
    # Paddle's PPStructure can handle a list of images or a PDF path
    # If it's a PDF, it returns a list of lists (one list per page)
    layout_results = layout_engine(file_path)
    
    final_structured_output = []

    # Iterate through pages (Paddle returns a list for each page)
    for page_index, page_regions in enumerate(layout_results):
        page_data = {"page": page_index + 1, "regions": []}
        
        # In a PDF context, we need the actual image of the page to show vLLM
        # Paddle stores the processed image in the result
        # Note: Depending on version, you might need to reload the image via fitz
        
        for region in page_regions:
            rtype = region['type'].lower()
            vlm_task = "table" if rtype == "table" else "ocr"
            
            # Call vLLM for this specific chunk
            # We use the 'Task:' prompt style recommended by Paddle
            prompt = f"Task: {vlm_task}"
            
            # Prepare mini-request for vLLM
            vllm_job_input = {
                "input": {
                    "prompt": prompt,
                    "image_url": raw_data, # Use the original URL/Base64
                    "sampling_params": job["input"].get("sampling_params", {"max_tokens": 512})
                }
            }
            
            content = ""
            async for chunk in vllm_engine.generate(JobInput(vllm_job_input)):
                content = chunk
            
            page_data["regions"].append({
                "type": rtype,
                "bbox": region['bbox'],
                "content": content
            })
            
        final_structured_output.append(page_data)

    # Cleanup temp file
    if os.path.exists(file_path):
        os.remove(file_path)

    yield {"output": final_structured_output}

runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": lambda x: vllm_engine.max_concurrency,
    "return_aggregate_stream": True,
})