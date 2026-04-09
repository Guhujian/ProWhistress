import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import json
from tqdm import tqdm
from transformers import WhisperConfig
# Import your project modules (adjust imports based on your actual directory structure)
from ..model.model import WhiStress
from .data_loader import load_data

def analyze_gate_on_test_set(args):
    # 1. Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Load model config and weights
    print(f"Loading model from {args.model_path}...")
    
    # Read metadata to restore configuration
    with open(os.path.join(args.model_path, "metadata.json"), "r") as f:
        meta = json.load(f)
    
    # Initialize model structure (must match training)
    config = WhisperConfig()
    model = WhiStress(
        config,
        d_ctx=meta.get("d_ctx", 256),
        stress_encoder_layers=meta.get("stress_encoder_layers", 2),
        stress_encoder_input_layer=meta.get("stress_encoder_input_layer", 12),
        decoder_input_layer=meta.get("decoder_input_layer", 12),
        whisper_backbone_name=args.whisper_path
    )
    
    # Load weights
    model.load_model(args.model_path, device=device)
    model.to(device)
    model.eval()

    # 3. Load test set
    print(f"Loading test dataset: {args.dataset_eval}...")
    dataset_obj = load_data(model, args.transcription_column_name, dataset_name=args.dataset_eval, save_path=args.dataset_path)
    _, _, test_dataset = dataset_obj.split_train_val_test()
    
    # Randomly sample a few examples, or specify indices
    # Here we simply take the first 20 samples for analysis
    num_samples_to_plot = 20
    indices = range(num_samples_to_plot)

    # 4. Register hook to capture gate values
    gate_activations = {} # Store gate values from the current forward pass
    
    def get_gate_hook(module, input, output):
        # output shape: [Batch, Seq_Len, Dim]
        # Take mean over feature dim to get [Batch, Seq_Len], representing
        # average gate strength for each token across all feature dimensions.
        gate_mean = output.mean(dim=-1).detach().cpu()
        gate_activations['current'] = gate_mean

    # Register on the Sigmoid layer (the last layer of gate_net)
    hook_handle = model.fusion_gate.gate_net[-1].register_forward_hook(get_gate_hook)

    # 5. Run inference in a loop and plot
    print(f"Analyzing first {num_samples_to_plot} samples...")
    os.makedirs(args.output_dir, exist_ok=True)

    for i in indices:
        sample = test_dataset[i]
        
        # Prepare inputs
        input_features = torch.tensor(sample["input_features"]).unsqueeze(0).to(device) # [1, 80, 3000]
        whisper_labels = torch.tensor(sample[f"labels_{args.transcription_column_name}"]).unsqueeze(0).to(device) # [1, Seq_Len]
        labels_head = torch.tensor(sample[f"labels_head_{args.transcription_column_name}"]).unsqueeze(0).to(device) # [1, Seq_Len]

        # To make forward pass work, construct decoder_input_ids
        # Whisper labels are usually padded with -100; convert them back to input_ids.
        decoder_input_ids = whisper_labels.clone()
        decoder_input_ids[decoder_input_ids == -100] = model.config.pad_token_id
        # Shift-right logic (Whisper typically needs a start token; simplified here by using labels as forcing)
        # Strictly speaking, we should shift right, but if labels are passed, the model handles shifting internally.
        
        with torch.no_grad():
            # Forward pass
            outputs = model(
                input_features=input_features,
                whisper_labels=whisper_labels, # Pass labels so model computes decoder_input_ids internally
                labels_head=labels_head
            )
            
            # Get predictions (0 or 1)
            preds = outputs.preds.cpu().numpy()[0] # [Seq_Len]
            
            # Get gate values
            gate_vals = gate_activations['current'][0].numpy() # [Seq_Len]

        # --- Post-processing and visualization ---
        
        # 1. Decode text (convert token IDs to tokens)
        # Filter out labels with -100
        valid_mask = whisper_labels[0].cpu() != -100
        valid_tokens = whisper_labels[0][valid_mask].cpu().tolist()
        text_tokens = [model.processor.tokenizer.decode([t]) for t in valid_tokens]
        
        # Align lengths: gate values, predictions, and labels should all use valid positions
        # Note: forward output length may match label length.
        seq_len = len(text_tokens)
        current_gate = gate_vals[:seq_len]
        current_preds = preds[:seq_len]
        current_labels = labels_head[0][valid_mask].cpu().numpy()

        # Filter special tokens
        ignore_tokens = {"<|startoftranscript|>", "<|endoftext|>", "<|notimestamps|>"}
        keep_indices = [i for i, t in enumerate(text_tokens) if t not in ignore_tokens]
        
        text_tokens = [text_tokens[i] for i in keep_indices]
        current_gate = current_gate[keep_indices]
        current_preds = current_preds[keep_indices]
        current_labels = current_labels[keep_indices]
        seq_len = len(text_tokens)

        # Save plotting data for potential combined visualization later
        data_to_save = {
            "tokens": text_tokens,
            "gate_values": current_gate.tolist(),
            "labels": current_labels.tolist(),
            "preds": current_preds.tolist()
        }
        json_save_path = os.path.join(args.output_dir, f"gate_analysis_sample_{i}_data.json")
        with open(json_save_path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        print(f"Saved data to {json_save_path}")

        # 2. Plot
        plt.figure(figsize=(15, 6))
        
        # Plot gate activation curve
        x = range(seq_len)
        plt.plot(x, current_gate, label='Gate Activation (Avg)', color='blue', linewidth=2, marker='o', markersize=4)
        
        # Mark ground-truth stress positions with red background spans
        for idx, label in enumerate(current_labels):
            if label == 1:
                plt.axvspan(idx - 0.4, idx + 0.4, color='red', alpha=0.3, label='GT Stress' if 'GT Stress' not in plt.gca().get_legend_handles_labels()[1] else "")

        # Mark predicted stress positions with green stars
        # stress_preds_indices = [idx for idx, p in enumerate(current_preds) if p == 1]
        # plt.scatter(stress_preds_indices, [current_gate[idx] for idx in stress_preds_indices], 
        #             color='green', marker='*', s=200, zorder=5, label='Predicted Stress')

        plt.xticks(x, text_tokens, rotation=45, ha='right', fontsize=18)
        plt.ylabel("Gate Value (0-1)", fontsize=18)
        plt.yticks(fontsize=16)
        # plt.title(f"Gate Mechanism Visualization - Sample {i}\n(Red Area = Ground Truth Stress)")
        plt.legend(loc='upper right', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(args.output_dir, f"gate_analysis_sample_{i}.pdf")
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
        plt.close()

    hook_handle.remove()
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to best_model folder")
    parser.add_argument("--dataset_eval", type=str, default="tinyStress-15K")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--transcription_column_name", type=str, default="transcription")
    parser.add_argument("--output_dir", type=str, default="analysis_plots")
    parser.add_argument("--whisper_path", type=str, default="./whistress/whisper_model/whisper-small.en", help="Path to local whisper model")
    args = parser.parse_args()
    
    analyze_gate_on_test_set(args)