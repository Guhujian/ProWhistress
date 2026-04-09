
"""
Script to download the Whisper model to a local directory.
"""

import os
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch

def download_whisper_model():
    """
    Download the whisper-small.en model and processor to a local directory.
    """
    model_name = "openai/whisper-small.en"
    local_model_dir = "./whistress/whisper_model/whisper-small.en"
    
    print(f"Downloading {model_name} model...")
    print(f"Save path: {local_model_dir}")
    
    # Create local directory
    os.makedirs(local_model_dir, exist_ok=True)
    
    try:
        # Download model
        print("Downloading model...")
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        model.save_pretrained(local_model_dir)
        print("✓ Model downloaded successfully")
        
        # Download processor
        print("Downloading processor...")
        processor = WhisperProcessor.from_pretrained(model_name)
        processor.save_pretrained(local_model_dir)
        print("✓ Processor downloaded successfully")
        
        # Display model info
        print(f"\nModel info:")
        print(f"- Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
        print(f"- Local path: {os.path.abspath(local_model_dir)}")
        
        # Verify model loads correctly
        print("\nVerifying local model...")
        test_model = WhisperForConditionalGeneration.from_pretrained(local_model_dir)
        test_processor = WhisperProcessor.from_pretrained(local_model_dir)
        print("✓ Local model verified successfully")
        
        print(f"\n✅ Download complete!")
        print(f"You can now load the model from: {local_model_dir}")
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = download_whisper_model()
    if success:
        print("\n🎉 All done!")
    else:
        print("\n💥 An error occurred during download")