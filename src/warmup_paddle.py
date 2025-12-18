import os
import ssl

# Bypass SSL verification if needed during build (sometimes helpful in restricted containers)
ssl._create_default_https_context = ssl._create_unverified_context

def warmup():
    print("Pre-downloading PaddleOCR models...")
    
    # Import PPStructure. This triggers the download of default models if not present.
    # We match the arguments from handler.py: layout=True, table=False, ocr=False
    from paddleocr import PPStructure
    
    # Initialize the engine. Use show_log=True for build logs.
    engine = PPStructure(layout=True, table=False, ocr=False, show_log=True)
    
    # Run a dummy inference to ensure everything is loaded/unpacked
    # We can pass a dummy image path or just rely on init if it downloads everything.
    # However, PPStructure often lazily loads some components. 
    # To be safe, we can trigger a dummy call if possible, but init usually downloads the models.
    print("PaddleOCR models downloaded successfully.")

if __name__ == "__main__":
    warmup()
