import os
import torch
import soundfile as sf
import numpy as np
from whistress import WhiStressInferenceClient

# Path to the input audio file
audio_path = "./test_audio/test.wav"

# Path to the trained model
model_path = "./train_results/Enc3_Input9_256/seed42/best_model"

# Initialize the model (load from best_model)
whistress_client = WhiStressInferenceClient(
    device="cuda" if torch.cuda.is_available() else "cpu",
    model_path=model_path,
)

# Load audio (soundfile returns a numpy array, no torchaudio needed)
waveform, sr = sf.read(audio_path, dtype="float32", always_2d=True)

# Convert to mono by averaging channels
waveform = waveform.mean(axis=1)

# Prepare input dict
audio_input = {
    "array": waveform,
    "sampling_rate": sr
}

# Run inference
pred_stress_pairs = whistress_client.predict(
    audio=audio_input,
    transcription=None,
    return_pairs=True
)

# Build output file path (same name as audio, .txt extension)
base_name = os.path.splitext(os.path.basename(audio_path))[0]
output_path = os.path.join(os.path.dirname(audio_path), f"{base_name}.txt")

# Write results to file
with open(output_path, "w", encoding="utf-8") as f:
    for char, stress in pred_stress_pairs:
        line = f"{char}\t{'[EMPHASIS]' if stress else ''}\n"
        print(line.strip())
        f.write(line)

print(f"\nResults saved to: {output_path}")
