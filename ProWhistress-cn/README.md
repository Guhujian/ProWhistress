# ProWhistress-cn

This repository contains the official Chinese implementation of **ProWhistress**, a dual-stream speech transcription architecture that augments the Whisper backbone with explicit acoustic modeling to effectively resolve the semantic-prosodic trade-off in alignment-free sentence stress detection.


## Getting Started

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### 📊 Dataset & Reproducibility
Due to double-blind constraints, the full **SinoStress** and **SinoStressReal** datasets will be released after acceptance.

### Training

```bash
python train_chinese.py \
  --dataset_path <path/to/dataset> \
  --output_path ./train_results/Enc3_Input9_256/seed42 \
  --transcription_column_name transcription \
  --d_ctx 256 \
  --stress_encoder_layers 3 \
  --layer_for_head 9 \
  --stress_encoder_input_layer 9 \
  --decoder_input_layer 12 \
  --dataset_train Aishell_final \
  --dataset_eval Aishell_final \
  --is_train True \
  --train_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --seed 42 \
  --stress_reg_coeff 0 \
  --dropout 0.0
```

### Evaluation

```bash
python train_chinese.py \
  --is_train False \
  --model_path ./train_results/Enc3_Input9_256/seed42/best_model \
  --dataset_path <path/to/dataset> \
  --dataset_eval SinoReal_TestOnly \
  --transcription_column_name transcription \
  --stress_encoder_layers 3 \
  --stress_encoder_input_layer 9 \
  --layer_for_head 9 \
  --decoder_input_layer 12 \
  --output_path ./eval_results/Enc3_Input9_256/seed42
```

### Gate Analysis

```powershell
python -m whistress.training.analyze_gate `
  --model_path ./train_results/Enc3_Input9_256/seed42/best_model `
  --dataset_path <path/to/dataset> `
  --dataset_eval SinoReal_TestOnly `
  --transcription_column_name transcription `
  --output_dir ./analysis_results/SinoReal/seed42
```

> **Note:** The training and evaluation commands use `\` as the line continuation character (Linux/macOS/Git Bash). The gate analysis command uses `` ` `` (Windows PowerShell).

---

### Inference

```bash
python inference_by_audio.py
```

Edit `inference_by_audio.py` to set the path to your audio file and model:

```python
audio_path = "./test_audio/test.wav"
model_path = "./train_results/Enc3_Input9_256/seed42/best_model"
```

Results (character + emphasis label) are written to `./test_audio/test.txt`.

---

## Arguments

| Argument | Description |
|----------|-------------|
| `--is_train` | Set to `True` for training mode, `False` for evaluation mode |
| `--dataset_train` | Name of the training dataset split |
| `--dataset_eval` | Name of the evaluation dataset split |
| `--dataset_path` | Local path to the dataset |
| `--output_path` | Path for saving model checkpoints and logs |
| `--model_path` | Path to a trained model for evaluation or gate analysis |
| `--transcription_column_name` | Column name for transcription text in the dataset |
| `--d_ctx` | Dimension of the context vector in the stress encoder |
| `--stress_encoder_layers` | Number of Transformer layers in the stress encoder |
| `--layer_for_head` | Whisper encoder layer index used as input to the classification head |
| `--stress_encoder_input_layer` | Whisper encoder layer index used as stress encoder input |
| `--decoder_input_layer` | Whisper decoder layer index used as additional decoder input |
| `--train_batch_size` | Training batch size per device |
| `--gradient_accumulation_steps` | Number of gradient accumulation steps |
| `--seed` | Random seed for reproducibility |
| `--stress_reg_coeff` | Coefficient for the stress regularization loss term |
| `--dropout` | Dropout probability |
| `--output_dir` | Output directory for gate analysis results |
