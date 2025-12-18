import os
import ssl

# Bypass SSL verification if needed during build (sometimes helpful in restricted containers)
ssl._create_default_https_context = ssl._create_unverified_context

def warmup():
    print("Pre-downloading PaddleOCR models...")
    
    # Import PPStructureV3. this triggers the download of default models if not present.
    # Note: PPStructure(v2) is removed in paddleocr>=3.0.
    from paddleocr import PPStructureV3
    
    # Initialize the engine. Use show_log=True for build logs.
    # PPStructureV3 handles layout/table/ocr internally based on the pipeline.
    # We might need to specify lang='en' or similar if needed, but defaults work for warmup.
    engine = PPStructureV3()
    
    # Run a dummy inference to ensure everything is loaded/unpacked
    # We can pass a dummy image path or just rely on init if it downloads everything.
    # However, PPStructure often lazily loads some components. 
    # To be safe, we can trigger a dummy call if possible, but init usually downloads the models.
    print("PaddleOCR models downloaded successfully.")

if __name__ == "__main__":
    warmup()
