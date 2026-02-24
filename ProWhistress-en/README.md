# ProWhistress-en

## Getting Started

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Training

```bash
python -m whistress.training.train \
  --is_train true \
  --dataset_train tinyStress-15K \
  --dataset_eval tinyStress-15K \
  --dataset_path <path/to/dataset> \
  --output_path <path/to/output> \
  --transcription_column_name transcription \
  --d_ctx 256 \
  --stress_encoder_layers 3 \
  --stress_encoder_input_layer 9 \
  --decoder_input_layer 12 \
  --train_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --seed 42 \
  --stress_reg_coeff 0 \
  --dropout 0.0 \
  --resume_from_checkpoint False
```

### Evaluation

```bash
python -m whistress.training.train \
  --is_train false \
  --model_path ./train_results/seed42/best_model \
  --dataset_path <path/to/dataset> \
  --dataset_eval tinyStress-15K \
  --transcription_column_name transcription \
  --stress_encoder_input_layer 9 \
  --output_path <path/to/output>
```

### Gate Analysis

```powershell
python -m whistress.training.analyze_gate `
  --model_path ./train_results/seed42/best_model `
  --dataset_path <path/to/dataset> `
  --dataset_eval tinyStress-15K `
  --transcription_column_name transcription `
  --output_dir <path/to/output/dir>
```

> **Note:** The training and evaluation commands use `\` as the line continuation character (Linux/macOS/Git Bash). The gate analysis command uses `` ` `` (Windows PowerShell).

---

## Arguments

| Argument | Description |
|----------|-------------|
| `--is_train` | Set to `true` for training mode, `false` for evaluation mode |
| `--dataset_train` | Name of the training dataset |
| `--dataset_eval` | Name of the evaluation dataset |
| `--dataset_path` | Local path to the dataset |
| `--output_path` | Path for saving model checkpoints and logs |
| `--model_path` | Path to a trained model for evaluation |
| `--transcription_column_name` | Column name for transcription text in the dataset |
| `--d_ctx` | Dimension of the context vector |
| `--stress_encoder_layers` | Number of layers in the stress encoder |
| `--stress_encoder_input_layer` | Whisper encoder layer index used as stress encoder input |
| `--decoder_input_layer` | Whisper decoder layer index used as decoder input |
| `--train_batch_size` | Training batch size |
| `--gradient_accumulation_steps` | Number of gradient accumulation steps |
| `--seed` | Random seed |
| `--stress_reg_coeff` | Stress regularization coefficient |
| `--dropout` | Dropout probability |
| `--resume_from_checkpoint` | Whether to resume training from a checkpoint |
| `--output_dir` | Output directory for gate analysis results |
