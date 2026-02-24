#!/usr/bin/env python3
"""
Standalone WhiStress model evaluation script.

Allows users to manually load a trained model after training and generate
evaluation results on the validation set. Supports both word-level and
token-level evaluation with detailed performance metrics.

Usage:
    python evaluate_model.py --model_path /path/to/trained/model --dataset_path /path/to/dataset

Example:
    python evaluate_model.py --model_path ./training_results_dctx256_9_encoder_seed43/best_model --dataset_path ./preprocessed_dataset
"""

import torch
import numpy as np
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from transformers import WhisperConfig, set_seed
from whistress.model.model import WhiStress
from whistress.training.data_loader import load_data
from whistress.training.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from whistress.training.trainer import WhiStressTrainer
from whistress.training.metrics import WhiStressMetrics
from transformers import Seq2SeqTrainingArguments

CURRENT_DIR = Path(__file__).parent

def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO
    )
    return logging.getLogger(__name__)

def load_model_from_path(model_path, device="cuda"):
    """
    Load a trained WhiStress model from the specified path.

    Args:
        model_path: Path to the saved model directory.
        device: Device to run on ("cuda" or "cpu").

    Returns:
        Loaded WhiStress model.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading model from: {model_path}")

    # Check that the model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    # Read model configuration from metadata file if available
    meta_path = os.path.join(model_path, "metadata.json")
    d_ctx = 256  # default
    stress_encoder_layers = 2  # default
    layer_for_head = 9  # default

    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            d_ctx = metadata.get("d_ctx", d_ctx)
            stress_encoder_layers = metadata.get("stress_encoder_layers", stress_encoder_layers)
            layer_for_head = metadata.get("layer_for_head", layer_for_head)
            logger.info(f"Config from metadata: d_ctx={d_ctx}, stress_encoder_layers={stress_encoder_layers}, layer_for_head={layer_for_head}")
        except Exception as e:
            logger.warning(f"Failed to read metadata.json: {e}. Using defaults.")

    # Initialise Whisper config and model
    whisper_backbone_name = str(CURRENT_DIR / "whistress" / "whisper_model" / "whisper-small.en")
    whisper_config = WhisperConfig()

    # Create model instance
    whistress_model = WhiStress(
        config=whisper_config,
        layer_for_head=layer_for_head,
        whisper_backbone_name=whisper_backbone_name,
        d_ctx=d_ctx,
        stress_encoder_layers=stress_encoder_layers
    )

    # Load model weights
    try:
        whistress_model.load_model(model_path)
        logger.info("Model weights loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
        raise

    # Move model to the specified device
    whistress_model = whistress_model.to(device)

    # Set to evaluation mode
    whistress_model.eval()

    return whistress_model

def evaluate_model(model, dataset, data_collator, output_dir, dataset_name="validation"):
    """
    Evaluate model performance.

    Args:
        model: Trained WhiStress model.
        dataset: Evaluation dataset.
        data_collator: Data collator.
        output_dir: Directory for saving results.
        dataset_name: Name of the dataset for logging.

    Returns:
        Dictionary of evaluation results.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting evaluation on dataset: {dataset_name}")

    # Create temporary training arguments (evaluation only)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=32,
        generation_max_length=96,
        dataloader_num_workers=0,  # avoid multiprocessing issues
        seed=42,
        fp16=False,  # disable mixed precision for evaluation stability
        do_train=False,
        do_eval=True,
    )

    # Set random seed
    set_seed(training_args.seed)

    # Initialise evaluation metrics
    metrics = WhiStressMetrics()

    # Create trainer (evaluation only)
    trainer = WhiStressTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset,  # required by trainer but not used
        eval_dataset=dataset,
        data_collator=data_collator,
        compute_metrics=metrics.compute_metrics,
        compute_loss_func=model.loss_fct,
        processing_class=model.processor.feature_extractor,
    )

    # Run word-level evaluation
    logger.info("Running word-level evaluation...")
    word_level_results = trainer.evaluate_at_word_level(
        ignore_keys=["whisper_logits"],
        eval_dataset=dataset,
        dataset_name=f"{dataset_name}-word_level",
    )

    # Run token-level evaluation
    logger.info("Running token-level evaluation...")
    token_level_results = trainer.evaluate(
        ignore_keys=["whisper_logits"],
        eval_dataset=dataset,
        dataset_name=f"{dataset_name}-token_level",
    )
    
    return {
        "word_level": word_level_results,
        "token_level": token_level_results
    }

def save_evaluation_results(results, output_dir, model_path, dataset_name):
    """
    Save evaluation results to files.

    Args:
        results: Dictionary of evaluation results.
        output_dir: Output directory.
        model_path: Path to the model.
        dataset_name: Name of the dataset.
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Prepare result data
    evaluation_summary = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "dataset_name": dataset_name,
        "results": results,
        "summary": {
            "word_level": {
                "accuracy": results["word_level"].get("eval_accuracy", 0),
                "precision": results["word_level"].get("eval_precision", 0),
                "recall": results["word_level"].get("eval_recall", 0),
                "f1": results["word_level"].get("eval_f1", 0),
            },
            "token_level": {
                "accuracy": results["token_level"].get("eval_accuracy", 0),
                "precision": results["token_level"].get("eval_precision", 0),
                "recall": results["token_level"].get("eval_recall", 0),
                "f1": results["token_level"].get("eval_f1", 0),
            }
        }
    }
    
    # Save detailed results
    results_file = os.path.join(output_dir, "evaluation_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_summary, f, indent=2, ensure_ascii=False)

    # Save concise summary
    summary_file = os.path.join(output_dir, "evaluation_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"Model Evaluation Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp:   {evaluation_summary['timestamp']}\n")
        f.write(f"Model path:  {model_path}\n")
        f.write(f"Dataset:     {dataset_name}\n\n")

        f.write(f"Word-level results:\n")
        f.write(f"  Accuracy:  {evaluation_summary['summary']['word_level']['accuracy']:.4f}\n")
        f.write(f"  Precision: {evaluation_summary['summary']['word_level']['precision']:.4f}\n")
        f.write(f"  Recall:    {evaluation_summary['summary']['word_level']['recall']:.4f}\n")
        f.write(f"  F1:        {evaluation_summary['summary']['word_level']['f1']:.4f}\n\n")

        f.write(f"Token-level results:\n")
        f.write(f"  Accuracy:  {evaluation_summary['summary']['token_level']['accuracy']:.4f}\n")
        f.write(f"  Precision: {evaluation_summary['summary']['token_level']['precision']:.4f}\n")
        f.write(f"  Recall:    {evaluation_summary['summary']['token_level']['recall']:.4f}\n")
        f.write(f"  F1:        {evaluation_summary['summary']['token_level']['f1']:.4f}\n")

    logger.info(f"Evaluation results saved to: {results_file}")
    logger.info(f"Evaluation summary saved to: {summary_file}")

    # Print summary to console
    print("\n" + "=" * 60)
    print("Model Evaluation Summary")
    print("=" * 60)
    print(f"Word-level  - F1: {evaluation_summary['summary']['word_level']['f1']:.4f}, "
          f"Precision: {evaluation_summary['summary']['word_level']['precision']:.4f}, "
          f"Recall: {evaluation_summary['summary']['word_level']['recall']:.4f}")
    print(f"Token-level - F1: {evaluation_summary['summary']['token_level']['f1']:.4f}, "
          f"Precision: {evaluation_summary['summary']['token_level']['precision']:.4f}, "
          f"Recall: {evaluation_summary['summary']['token_level']['recall']:.4f}")
    print("=" * 60)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Standalone WhiStress model evaluation script")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model (e.g. ./training_results_dctx256_3_encoder/best_model)"
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./preprocessed_dataset",
        help="Path to the preprocessed dataset directory (default: ./preprocessed_dataset)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for evaluation results (default: ./evaluation_results_<timestamp>)"
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["tinyStress-15K"],
        default="tinyStress-15K",
        help="Dataset name (default: tinyStress-15K)"
    )

    parser.add_argument(
        "--transcription_column_name",
        type=str,
        choices=["transcription", "aligned_whisper_transcriptions"],
        default="transcription",
        help="Transcription column name (default: transcription)"
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=["test", "validation", "train"],
        default="test",
        help="Dataset split to evaluate (default: test)"
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to use (default: auto)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    logger.info(f"Using device: {device}")

    # Set random seed
    set_seed(args.seed)
    logger.info(f"Random seed: {args.seed}")

    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"./evaluation_results_{timestamp}"

    try:
        # Load model
        logger.info("Loading model...")
        model = load_model_from_path(args.model_path, device)
        logger.info("Model loaded successfully")

        # Load dataset
        logger.info("Loading dataset...")
        train_dataset, val_dataset, test_dataset = load_data(
            model_with_emphasis_head=model,
            transcription_column_name=args.transcription_column_name,
            dataset_name=args.dataset_name,
            save_path=args.dataset_path
        )

        # Select the desired split
        if args.split == "test":
            eval_dataset = test_dataset
        elif args.split == "validation":
            eval_dataset = val_dataset
        elif args.split == "train":
            eval_dataset = train_dataset
        else:
            raise ValueError(f"Unsupported dataset split: {args.split}")

        logger.info(f"Dataset loaded. Split: {args.split}, samples: {len(eval_dataset)}")

        # Create data collator
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=model.processor,
            decoder_start_token_id=model.config.decoder_start_token_id,
        )

        # Run evaluation
        logger.info("Starting evaluation...")
        results = evaluate_model(
            model=model,
            dataset=eval_dataset,
            data_collator=data_collator,
            output_dir=args.output_dir,
            dataset_name=f"{args.dataset_name}-{args.split}"
        )

        # Save results
        save_evaluation_results(
            results=results,
            output_dir=args.output_dir,
            model_path=args.model_path,
            dataset_name=f"{args.dataset_name}-{args.split}"
        )

        logger.info("Evaluation complete!")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    main()