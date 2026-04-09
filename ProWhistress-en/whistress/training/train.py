# train.py is the main entry point for WhiStress training,
# including training, evaluation, and argument parsing.
import torch
import numpy as np
import os
import sys
import logging
import json
import argparse
from pathlib import Path
from transformers import WhisperConfig, Seq2SeqTrainingArguments, TrainerCallback, set_seed, EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint
from ..model.model import WhiStress
from .data_loader import load_data
from .data_collator import DataCollatorSpeechSeq2SeqWithPadding
from .trainer import WhiStressTrainer
from .metrics import WhiStressMetrics

CURRENT_DIR = Path(__file__).parent
WANDB_API_KEY = os.environ.get("WAND_API_KEY", None)

# Custom callback with extra save-time handling for special types.
class CustomCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        # Use a dedicated logger to ensure output is visible in terminal logs.
        self.logger = logging.getLogger("whistress.training.train")
        # Keep level low enough to show INFO messages.
        self.logger.setLevel(logging.INFO)
        # Rely on Transformers' built-in best-metric tracking.
    
    def on_save(self, args, state, control, **kwargs):
        # Example of handling serialization
        if hasattr(state, "metrics") and isinstance(state.metrics, np.ndarray):
            state.metrics = state.metrics.tolist()
    
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """Log metrics after each evaluation; best-model decisions are handled by Transformers."""
        # Debug output
        print(f"🔍 CustomCallback.on_evaluate called - step: {state.global_step}")
        sys.stdout.flush()  # Force flush stdout buffer
        
        # Support metrics argument (Transformers usually passes metrics, not logs)
        if logs is None and "metrics" in kwargs:
            logs = kwargs["metrics"]
        
        if logs is None:
            print("⚠️ logs is None, skipping evaluation")
            sys.stdout.flush()
            return
            
        # Get the current evaluation metric
        current_metric_key = args.metric_for_best_model
        print(f"🎯 Looking for metric: {current_metric_key}")
        print(f"📋 Available metrics: {list(logs.keys())}")
        sys.stdout.flush()
        
        if current_metric_key not in logs:
            print(f"❌ Metric {current_metric_key} is not present in logs")
            sys.stdout.flush()
            return
            
        current_metric = logs[current_metric_key]
        print(f"📊 Current {current_metric_key}: {current_metric:.4f}")
        
        # Use Transformers built-in best-model tracking
        # Check whether state.best_metric exists and has been updated
        state_best_metric = getattr(state, 'best_metric', None)
        print(f"🏆 Transformers tracked best metric: {state_best_metric}")
        sys.stdout.flush()
        
        # Check whether this is a new best model using Transformers logic
        is_better = False
        if state_best_metric is None:
            is_better = True
            print(f"🆕 First evaluation, set as current best model")
        else:
            # Compare current metric with Transformers tracked best metric
            if args.greater_is_better:
                is_better = current_metric > state_best_metric
            else:
                is_better = current_metric < state_best_metric
            print(f"🔄 Compare: current={current_metric:.4f} vs transformers_best={state_best_metric:.4f}, better={is_better}")
        sys.stdout.flush()
        
        if is_better:
            print(f"🎉 New best model found! {current_metric_key}: {current_metric:.4f}")
            print(f"📊 Current evaluation metrics - Accuracy: {logs.get('eval_accuracy', 'N/A'):.4f}, "
                           f"Precision: {logs.get('eval_precision', 'N/A'):.4f}, "
                           f"Recall: {logs.get('eval_recall', 'N/A'):.4f}, "
                           f"F1: {logs.get('eval_f1', 'N/A'):.4f}")
            sys.stdout.flush()
            
            self.logger.info(f"🎉 New best model found! {current_metric_key}: {current_metric:.4f}")
            self.logger.info(f"📊 Current evaluation metrics - Accuracy: {logs.get('eval_accuracy', 'N/A'):.4f}, "
                           f"Precision: {logs.get('eval_precision', 'N/A'):.4f}, "
                           f"Recall: {logs.get('eval_recall', 'N/A'):.4f}, "
                           f"F1: {logs.get('eval_f1', 'N/A'):.4f}")
        else:
            print(f"📈 Current model did not beat the best model (Transformers best: {state_best_metric:.4f})")
            sys.stdout.flush()

# Main training/evaluation flow
# args: parsed command-line arguments
def train_or_evaluate(args):
    # Select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    whisper_backbone_name = str(CURRENT_DIR.parent / "whisper_model" / "whisper-small.en")
    whisper_config = WhisperConfig()
    layer_for_head = 9
    
    # Print model configuration
    msg = "\n" + "="*60 + "\n"
    msg += "🔧 Model Configuration\n"
    msg += "="*60 + "\n"
    msg += f"   - Whisper Decoder extraction layer: {layer_for_head}\n"
    msg += f"   - StressEncoder input layer: {args.stress_encoder_input_layer}\n"
    msg += f"   - Additional Decoder input layer: {args.decoder_input_layer}\n"
    msg += f"   - StressEncoder regularization coefficient: {args.stress_reg_coeff}\n"
    msg += f"   - Random seed: {args.seed}\n"
    msg += "="*60 + "\n"
    print(msg)
    sys.stdout.flush()
    logger.info("Model Configuration:\n" + msg)

    # Load model or initialize a new one
    if args.model_path:
        logger.info(f"Loading model from {args.model_path}")
        # Try loading layer_for_head and d_ctx from saved metadata
        meta_path = os.path.join(args.model_path, "metadata.json")
        d_ctx_meta = args.d_ctx
        stress_encoder_input_layer_meta = args.stress_encoder_input_layer
        stress_encoder_layers_meta = args.stress_encoder_layers
        decoder_input_layer_meta = args.decoder_input_layer
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    metadata = json.load(f)
                layer_for_head = metadata.get("layer_for_head", layer_for_head)
                d_ctx_meta = metadata.get("d_ctx", d_ctx_meta)
                stress_encoder_layers_meta = metadata.get("stress_encoder_layers", stress_encoder_layers_meta)
                stress_encoder_input_layer_meta = metadata.get("stress_encoder_input_layer", stress_encoder_input_layer_meta)
                decoder_input_layer_meta = metadata.get("decoder_input_layer", decoder_input_layer_meta)
            except Exception as e:
                logger.warning(f"Failed to read metadata.json: {e}. Falling back to defaults.")
        whistress_model = WhiStress(
            whisper_config,
            layer_for_head=layer_for_head,
            whisper_backbone_name=whisper_backbone_name,
            d_ctx=d_ctx_meta,
            stress_encoder_layers=stress_encoder_layers_meta,
            stress_encoder_input_layer=stress_encoder_input_layer_meta,
            decoder_input_layer=decoder_input_layer_meta,
            stress_reg_coeff=args.stress_reg_coeff,
            dropout=args.dropout,
            freeze_stress_encoder=args.freeze_stress_encoder,
        ).to(device)
        whistress_model.load_model(args.model_path, device=device)
        whistress_model.to(device)
        whistress_model.eval()
    else:
        logger.info("Training a new model from scratch")
        whistress_model = WhiStress(
            whisper_config,
            layer_for_head=layer_for_head,
            whisper_backbone_name=whisper_backbone_name,
            d_ctx=args.d_ctx,
            stress_encoder_layers=args.stress_encoder_layers,
            stress_encoder_input_layer=args.stress_encoder_input_layer,
            decoder_input_layer=args.decoder_input_layer,
            stress_reg_coeff=args.stress_reg_coeff,
            dropout=args.dropout,
            freeze_stress_encoder=args.freeze_stress_encoder,
        ).to(device)
    
    # Set tokenizer input names
    whistress_model.processor.tokenizer.model_input_names = [
        "input_ids",
        "attention_mask",
        "labels_head",
    ]
    
    # Build data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=whistress_model.processor,
        decoder_start_token_id=whistress_model.whisper_model.config.decoder_start_token_id,
        forced_decoder_ids=whistress_model.whisper_model.config.forced_decoder_ids[
            0
        ][1],
        eos_token_id=whistress_model.whisper_model.config.eos_token_id,
        transcription_column_name=args.transcription_column_name
    )

    train, val = None, None
    # Load train and validation sets
    if args.is_train:
        DatasetTrain = load_data(whistress_model, args.transcription_column_name, dataset_name=args.dataset_train, save_path=args.dataset_path)
        train, val, _ = DatasetTrain.split_train_val_test()
    # Load test set
    DatasetEval = load_data(whistress_model, args.transcription_column_name, dataset_name=args.dataset_eval, save_path=args.dataset_path)
    _, _, test = DatasetEval.split_train_val_test()
    
    print(f"Output path for the training run: {args.output_path}")
    output_path = args.output_path

    # Weights & Biases logging
    if WANDB_API_KEY and args.is_train:
        import wandb
        wandb.login(key=WANDB_API_KEY)
        wandb.init(
            project="whistress",
            name=f"{args.dataset_train}_{args.dataset_eval}",
            config={
                "dataset_train": args.dataset_train,
                "dataset_eval": args.dataset_eval,
                "transcription_column_name": args.transcription_column_name,
            },
            dir=output_path,
        )

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_path,  # change to a repo name of your choice
        # per_device_train_batch_size=4, # assuming 8 gpus. decrease to ~2 for small dataset, increase to ~4 for large dataset.
        per_device_train_batch_size=args.train_batch_size, 
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # increase by 2x for every 2x decrease in batch size
        learning_rate=5e-4, # decrease to ~5e-4 for large dataset, increase to ~4e-4 for small dataset
        # warmup_ratio=0.05,
        warmup_ratio=0.05,
        num_train_epochs=2, # Increase to 4 for task-specific evaluation, use 2 for zero-shot - More generalized evaluation. 
        seed=args.seed,
        gradient_checkpointing=False,
        fp16=False,  # Enable mixed precision if needed to reduce memory and speed up training.
        do_train=True,
        do_eval=True,  
        eval_strategy="steps",
        save_strategy="steps",
        # per_device_eval_batch_size=4,
        per_device_eval_batch_size=args.eval_batch_size,
        generation_max_length=96,
        save_steps=100,
        save_total_limit=1,  # Keep only the most recent checkpoint(s).
        load_best_model_at_end=True,  # Load the best model at the end of training.
        metric_for_best_model="eval_f1",  # Use validation F1 as best-model criterion.
        greater_is_better=True,  # For F1, higher is better.
        eval_steps=10,
        dataloader_num_workers=8,
        logging_steps=10,
        weight_decay=0.01,
        push_to_hub=False,
        report_to=['wandb'] if WANDB_API_KEY is not None else None,
        label_names=[f"labels_head_{args.transcription_column_name}", "sentence_index", f"labels_{args.transcription_column_name}"],
        overwrite_output_dir=False, # change if you want to keep previous output
    )
    # Set random seed
    set_seed(training_args.seed)
    metrics = WhiStressMetrics()

    # Print training configuration
    if args.is_train:
        print("\n" + "="*60)
        print("🔧 Training Configuration")
        print("="*60)
        print(f"📁 Output directory: {args.output_path}")
        print(f"🎯 Training dataset: {args.dataset_train}")
        print(f"📊 Evaluation dataset: {args.dataset_eval}")
        print(f"📝 Transcription column: {args.transcription_column_name}")
        print(f"🧠 Model settings:")
        print(f"   - Whisper model: {whisper_backbone_name}")
        print(f"   - StressEncoder input layer: {args.stress_encoder_input_layer}")
        print(f"   - Additional Decoder input layer: {args.decoder_input_layer}")
        print(f"   - Whisper Decoder extraction layer: {layer_for_head}")
        print(f"   - Decoder context length: {args.d_ctx}")
        print(f"   - StressEncoder layers: {args.stress_encoder_layers}")
        
        # Calculate model parameter counts
        whistress_model.train()
        total_params = sum(p.numel() for p in whistress_model.parameters())
        trainable_params = sum(p.numel() for p in whistress_model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"📊 Model parameters:")
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")
        print(f"   - Frozen parameters: {frozen_params:,}")
        print(f"   - Trainable ratio: {trainable_params/total_params*100:.2f}%")
        print(f"⚙️ Training arguments:")
        print(f"   - Train batch size: {training_args.per_device_train_batch_size}")
        print(f"   - Eval batch size: {training_args.per_device_eval_batch_size}")
        print(f"   - Learning rate: {training_args.learning_rate}")
        print(f"   - Num epochs: {training_args.num_train_epochs}")
        print(f"   - Random seed: {training_args.seed}")
        print(f"   - Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
        print(f"   - Weight decay: {training_args.weight_decay}")
        print(f"   - Warmup ratio: {training_args.warmup_ratio}")
        print(f"   - Save steps: {training_args.save_steps}")
        print(f"   - Eval steps: {training_args.eval_steps}")
        print(f"   - Max generation length: {training_args.generation_max_length}")
        print(f"🎲 Dataset info:")
        if train is not None:
            print(f"   - Train set size: {len(train)}")
        if val is not None:
            print(f"   - Validation set size: {len(val)}")
        if test is not None:
            print(f"   - Test set size: {len(test)}")
        print("="*60)
        print()

    trainer_emphasis = None
    if args.is_train:
        # Build trainer
        trainer_emphasis = WhiStressTrainer(
        args=training_args,
        model=whistress_model,
        train_dataset=train,
        eval_dataset=val,
        data_collator=data_collator,
        compute_metrics=metrics.compute_metrics,
        compute_loss_func=whistress_model.loss_fct,
        processing_class=whistress_model.processor.feature_extractor,
        callbacks=[CustomCallback(), EarlyStoppingCallback(early_stopping_patience=20)],  # Add callback to monitor best model
        )
        # Pre-training evaluation
        # trainer_emphasis.evaluate_at_word_level(
        #     # to ignore whisper_logits in the compute_metrics function (only the custom head logits are used)
        #     ignore_keys=["whisper_logits"],
        #     eval_dataset=test,
        #     dataset_name=f"{args.dataset_eval}-initial-word_level",
        # )
        # trainer_emphasis.evaluate(
        #     # to ignore whisper_logits in the compute_metrics function (only the custom head logits are used)
        #     ignore_keys=["whisper_logits"],
        #     eval_dataset=test,
        #     dataset_name=f"{args.dataset_eval}-initial",
        # )
        # Start training
        print("🚀 Starting training...")
        resume_from_checkpoint = args.resume_from_checkpoint
        print(f"DEBUG: args.resume_from_checkpoint = {args.resume_from_checkpoint}")
        if resume_from_checkpoint is not None:
             if resume_from_checkpoint.lower() == "true":
                 resume_from_checkpoint = True
             elif resume_from_checkpoint.lower() == "false":
                 resume_from_checkpoint = None
        
        # Automatically detect checkpoint
        if resume_from_checkpoint is None and os.path.isdir(args.output_path):
            last_checkpoint = get_last_checkpoint(args.output_path)
            if last_checkpoint is not None:
            print(f"🔍 Checkpoint detected automatically: {last_checkpoint}. Training will resume from it.")
                resume_from_checkpoint = last_checkpoint
                 
        print(f"DEBUG: Calling train(resume_from_checkpoint={resume_from_checkpoint})")
        
        trainer_emphasis.train(resume_from_checkpoint=resume_from_checkpoint)
        print("✅ Training completed!")
        
        # Save final model
        print("💾 Saving final trained model...")
        trainer_emphasis.save_final_model(args.output_path, training_args)
        
        # If load_best_model_at_end is enabled, also save a copy of the best model
        if training_args.load_best_model_at_end:
            best_model_path = os.path.join(args.output_path, "best_model")
            os.makedirs(best_model_path, exist_ok=True)
            print(f"💎 Saving a copy of the best model to: {best_model_path}")
            trainer_emphasis.save_final_model(best_model_path, training_args)
    else:
        # Evaluation-only mode
        trainer_emphasis = WhiStressTrainer(
            args=training_args,
            model=whistress_model,
            train_dataset=test, # we don't really use it, but the trainer requires it
            eval_dataset=test, # we don't really use it, but the trainer requires it
            data_collator=data_collator,
            compute_metrics=metrics.compute_metrics,
            tokenizer=whistress_model.processor.feature_extractor,
        )

    # Post-training evaluation
    trainer_emphasis.evaluate_at_word_level(
        # to ignore whisper_logits in the compute_metrics function (only the custom head logits are used)
        ignore_keys=["whisper_logits"],
        eval_dataset=test,
        dataset_name=f"{args.dataset_eval}-final-word_level",
    )
    # trainer_emphasis.evaluate(
    #     # to ignore whisper_logits in the compute_metrics function (only the custom head logits are used)
    #     ignore_keys=["whisper_logits"],
    #     eval_dataset=test,
    #     dataset_name=f"{args.dataset_eval}-final",
    # )

# String-to-bool helper

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


# Main function to execute the training
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model to be loaded for evaluation.\
                If training, a model path (of the final model) must not be provided",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to the dataset directory to save to or load the preprocessed dataset from.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to the output directory for the training run. \
                If not provided, a new directory named training_results will be created under the current directory.",
    )
    parser.add_argument(
        "--transcription_column_name",
        type=str,
        choices=["transcription", "aligned_whisper_transcriptions"],
        default="transcription",
        help="""Name of the transcription column in the dataset. 
        transcription: The original transcription text as written from the raw dataset.
        aligned_whisper_transcriptions: The transcription text aligned with Whisper's output (small syntactic formulation differences).
    """,
    )
    parser.add_argument(
        "--d_ctx",
        type=int,
        default=256,
        help="Context bottleneck dimension for audio-text cross-attention",
    )
    parser.add_argument(
        "--stress_encoder_layers",
        type=int,
        default=2,
        help="Number of Transformer Encoder layers in StressEncoder (default: 2)",
    )
    parser.add_argument(
        "--stress_encoder_input_layer",
        type=int,
        default=12,
        help="Which Whisper Encoder layer output to use as input for StressEncoder (default: 12)",
    )
    parser.add_argument(
        "--decoder_input_layer",
        type=int,
        default=12,
        help="Which Whisper Encoder layer output to use as input for Additional Decoder Cross Attention (default: 12)",
    )
    parser.add_argument(
        "--dataset_train",
        type=str,
        choices=["tinyStress-15K"], # add other datasets as needed
        default="tinyStress-15K",
        help="Name of the dataset to be used for training and validation",
    )
    parser.add_argument(
        "--dataset_eval",
        type=str,
        choices=["tinyStress-15K"], # add other datasets as needed
        default="tinyStress-15K",
        help="Name of the dataset to be used for evaluation",
    )
    parser.add_argument(
        "--is_train",
        type=str2bool,
        default=True,
        help="Whether to train the model (True) or only evaluate it (False)",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="The batch size per device for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="The batch size per device for evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to control stochastic behavior during training.",
    )
    parser.add_argument(
        "--stress_reg_coeff",
        type=float,
        default=0.0,
        help="L2 regularization coefficient for StressEncoder output (default: 0.0)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.15,
        help="Dropout probability for StressEncoder output (default: 0.15)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=str2bool,
        default=True,
        help="If True, use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If specified, resumes training from this checkpoint path. Can be a path or 'True' to resume from the latest checkpoint in output_dir.",
    )
    parser.add_argument(
        "--freeze_stress_encoder",
        type=str2bool,
        default=False,
        help="Whether to freeze the StressEncoder during training (Linear Probing).",
    )
    
    args = parser.parse_args()

    # Auto-create output directory if not specified
    if not args.output_path:
        print("No output path provided, creating a new directory named 'training_results' in the current directory.")
        output_path = CURRENT_DIR / "training_results"
        output_path.mkdir(parents=True, exist_ok=True)
        args.output_path = str(output_path)

    if args.is_train:
        assert args.model_path is None, "If training, a model path (of the final model) must not be provided"
    else:
        assert args.model_path is not None, "If not training, a model (of the final model) path must be provided"
    
    train_or_evaluate(args)
