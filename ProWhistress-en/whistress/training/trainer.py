# trainer.py implements a custom trainer for the WhiStress model,
# supporting token-level/word-level evaluation and model saving.
from transformers import Seq2SeqTrainer
from tqdm import tqdm
import torch
import numpy as np
import os
import json

# Custom trainer extending transformers Seq2SeqTrainer.
class WhiStressTrainer(Seq2SeqTrainer):
    """
    Custom trainer extending Seq2SeqTrainer for speech emphasis detection.
    
    Implements specialized training, evaluation, and model saving methods
    designed specifically for the emphasis detection model architecture.
    """

    def _pad_tensors_to_max_len(self, tensor, max_length):
        """
        Pad tensors to a specified maximum length using -100 as padding token.
        
        Args:
            tensor: Input tensor to pad
            max_length: Target length for padded tensor
            
        Returns:
            Padded tensor of shape (batch_size, max_length)
        """
        # Pad to the target length using -100.
        pad_token_id = -100

        # Create a padded tensor using the custom pad token
        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )

        # Ensure that the tensor fits within the padded tensor up to the original tensor's length
        padded_tensor[:, : tensor.shape[-1]] = tensor

        return padded_tensor

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Execute a single training step with gradient clipping.
        
        Removes sentence indices from inputs before passing to parent class,
        then applies gradient clipping to prevent exploding gradients.
        
        Args:
            model: Model to train
            inputs: Dictionary of input tensors
            num_items_in_batch: Optional parameter specifying batch size
            
        Returns:
            Loss value for the training step
        """
        # Remove sentence_index during training so it does not affect gradients.
        sentence_index = inputs.pop("sentence_index")
        # Perform the default training step
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        # Clip gradients manually
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

        return loss

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time=None, **kwargs):
        """
        Use the built-in Transformers evaluation and saving flow.
        Delegate directly to the parent class to preserve official best-model logic.
        """
        return super()._maybe_log_save_evaluate(
            tr_loss,
            grad_norm,
            model,
            trial,
            epoch,
            ignore_keys_for_eval,
            start_time=start_time,
            **kwargs,
        )

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Save checkpoint and print brief status; best-model selection is handled by parent class.
        """
        print(f"💾 Starting checkpoint save - step: {self.state.global_step}")
        # Call parent save method (do not pass metrics because parent does not accept it).
        super()._save_checkpoint(model, trial)
        print(f"✅ Checkpoint save completed - step: {self.state.global_step}")

        # Print currently tracked best-model info (updated by parent class).
        best_metric = getattr(self.state, "best_metric", None)
        best_ckpt = getattr(self.state, "best_model_checkpoint", None)
        if best_metric is not None:
            try:
                print(f"🏆 Current best {self.args.metric_for_best_model}: {float(best_metric):.4f}")
            except Exception:
                print(f"🏆 Current best {self.args.metric_for_best_model}: {best_metric}")
        if best_ckpt:
            print(f"📂 Best model checkpoint: {best_ckpt}")

    def save_final_model(self, output_dir=None, training_args=None):
        """
        Save only the emphasis detection components of the model.
        
        Rather than saving the entire model, this method saves only:
        1. The classifier (head) used for emphasis detection
        2. The additional decoder block
        3. The selected layer passed to the head
        4. Training arguments for reproducibility
        
        Args:
            output_dir: Directory to save model components
            training_args: Training arguments to save
        """
        # Save only emphasis-detection related heads and additional decoder layers.
        classifier = (
            self.model.classifier if hasattr(self.model, "classifier") else None
        )
        additional_decoder_block = (
            self.model.additional_decoder_block
            if hasattr(self.model, "additional_decoder_block")
            else None
        )
        if output_dir is not None:
            print(f"💾 Saving final model to: {output_dir}")
            
            torch.save(
                classifier.state_dict(), os.path.join(output_dir, "classifier.pt")
            )
            torch.save(
                additional_decoder_block.state_dict(),
                os.path.join(output_dir, "additional_decoder_block.pt"),
            )
            # Also save stress-encoder branch
            if hasattr(self.model, "stress_encoder") and self.model.stress_encoder is not None:
                torch.save(
                    self.model.stress_encoder.state_dict(),
                    os.path.join(output_dir, "stress_encoder.pt"),
                )
            # Also save trainable audio feature extractor (if present)
            if hasattr(self.model, "audio_feature_extractor") and self.model.audio_feature_extractor is not None:
                torch.save(
                    self.model.audio_feature_extractor.state_dict(),
                    os.path.join(output_dir, "audio_feature_extractor.pt"),
                )
            # Also save fusion gate module (if present)
            if hasattr(self.model, "fusion_gate") and self.model.fusion_gate is not None:
                torch.save(
                    self.model.fusion_gate.state_dict(),
                    os.path.join(output_dir, "fusion_gate.pt"),
                )
            # Also save bottleneck linear layers (if present)
            if hasattr(self.model, "dec_to_ctx"):
                torch.save(
                    self.model.dec_to_ctx.state_dict(),
                    os.path.join(output_dir, "dec_to_ctx.pt"),
                )
            if hasattr(self.model, "enc_to_ctx"):
                torch.save(
                    self.model.enc_to_ctx.state_dict(),
                    os.path.join(output_dir, "enc_to_ctx.pt"),
                )
            if hasattr(self.model, "ctx_to_model"):
                torch.save(
                    self.model.ctx_to_model.state_dict(),
                    os.path.join(output_dir, "ctx_to_model.pt"),
                )
            # save the layer passed to the head and d_ctx into metadata
            layer_for_head = self.model.layer_for_head
            
            # Fix for WhisperEncoder which doesn't have .encoder attribute
            stress_encoder_layers = 2
            if hasattr(self.model, "stress_encoder"):
                if hasattr(self.model.stress_encoder, "config") and hasattr(self.model.stress_encoder.config, "encoder_layers"):
                     stress_encoder_layers = self.model.stress_encoder.config.encoder_layers
                elif hasattr(self.model.stress_encoder, "encoder"):
                     if hasattr(self.model.stress_encoder.encoder, "layers"):
                         stress_encoder_layers = len(self.model.stress_encoder.encoder.layers)
                     elif hasattr(self.model.stress_encoder.encoder, "num_layers"):
                         stress_encoder_layers = self.model.stress_encoder.encoder.num_layers
            
            metadata = {
                "layer_for_head": layer_for_head, 
                "d_ctx": getattr(self.model, "d_ctx", None),
                "stress_encoder_layers": stress_encoder_layers,
                "stress_encoder_input_layer": getattr(self.model, "stress_encoder_input_layer", 12),
                "decoder_input_layer": getattr(self.model, "decoder_input_layer", 12)
            }
            with open(os.path.join(output_dir, "metadata.json"), "w") as file:
                json.dump(metadata, file)
            # save the training arguments
            with open(os.path.join(output_dir, "training_args.json"), "w") as file:
                json.dump(training_args.to_dict(), file)
            
            print(f"✅ Model components saved successfully to: {output_dir}")
            print(f"   - classifier.pt")
            print(f"   - additional_decoder_block.pt")
            if hasattr(self.model, "stress_encoder") and self.model.stress_encoder is not None:
                print(f"   - stress_encoder.pt")
            if hasattr(self.model, "audio_feature_extractor") and self.model.audio_feature_extractor is not None:
                print(f"   - audio_feature_extractor.pt")
            if hasattr(self.model, "fusion_gate") and self.model.fusion_gate is not None:
                print(f"   - fusion_gate.pt")
            if hasattr(self.model, "dec_to_ctx"):
                print(f"   - dec_to_ctx.pt")
            if hasattr(self.model, "enc_to_ctx"):
                print(f"   - enc_to_ctx.pt")
            if hasattr(self.model, "ctx_to_model"):
                print(f"   - ctx_to_model.pt")
            print(f"   - metadata.json")
            print(f"   - training_args.json")
            
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", dataset_name=''):
        """
        Evaluate model at token level on the evaluation dataset.
        
        Runs a forward pass through the model for each batch, collects predictions
        and labels, and calculates evaluation metrics. Operates at the token level,
        meaning each token's emphasis prediction is evaluated separately.
        
        Args:
            eval_dataset: Dataset to evaluate on
            ignore_keys: Keys to ignore in the model output
            metric_key_prefix: Prefix for metric keys in output
            dataset_name: Name of the dataset for logging purposes
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Token-level evaluation: run batch inference and collect all predictions/labels.
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()

        all_preds = []
        all_labels = []

        for batch in tqdm(eval_dataloader):
            # Extract input features and labels
            input_features = batch["input_features"]
            labels_keys = [elem for i, elem in enumerate(batch.keys()) if "labels" in elem and not "labels_head" in elem][0]
            whisper_labels = batch[labels_keys]
            labels_head_keys = [elem for i, elem in enumerate(batch.keys()) if "labels_head" in elem][0]
            labels_head = batch[labels_head_keys]

            # Generate predictions by a forward pass through the model
            with torch.no_grad():
                """uncomment the following block if you want to use the generate method"""
                # generated_ids = self.model.generate(
                #     input_features=input_features,
                #     # decoder_input_ids=batch.get("decoder_input_ids"),  # If required
                #     max_length=self.args.generation_max_length,
                #     # labels_head=None,
                #     whisper_labels=whisper_labels,
                #     # **self.args.generation_kwargs
                # )
                # Run forward pass through the model to get predictions 
                # (assuming whisper labels are aligned in length with labels_head)
                # Instead of using generate(), we use a direct forward pass which is faster
                # and returns logits that can be converted to binary predictions
                generated_ids = self.model(
                    input_features=input_features,
                    labels_head=labels_head,
                    whisper_labels=whisper_labels
                )['preds']  # Extract the 'preds' field which contains binary predictions
                
            # Pad predictions and labels to the same length for proper comparison
            # This ensures all tensors in the batch have consistent dimensions
            padded_preds = self._pad_tensors_to_max_len(
                generated_ids, max_length=self.args.generation_max_length
            )
            padded_labels = self._pad_tensors_to_max_len(
                labels_head, max_length=self.args.generation_max_length
            )
            
            # Convert to CPU and numpy for accumulation
            # We collect all batch predictions before computing metrics
            all_preds.append(padded_preds.cpu().numpy())
            all_labels.append(padded_labels.cpu().numpy())

        # Concatenate all batches into single arrays for metric computation
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Compute evaluation metrics using the provided compute_metrics function
        # This typically calculates precision, recall, F1, and other relevant metrics
        outputs_metrics = {}
        if self.compute_metrics is not None:
            # The compute_metrics function expects a dictionary with predictions and labels
            metrics = self.compute_metrics(
                {"predictions": all_preds, "label_ids": all_labels}
            )
            for key, value in metrics.items():
                key = f"{metric_key_prefix}_{key}"
                if isinstance(value, np.ndarray):
                    outputs_metrics[key] = value.tolist()
                else:
                    outputs_metrics[key] = value
                    
        with open(os.path.join(self.args.output_dir, "log_eval.txt"), "a") as file:
            json.dump(f'Evaluate at TOKEN LEVEL {dataset_name}:', file)
            json.dump(outputs_metrics, file)
        self.log(outputs_metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, outputs_metrics)
        # print(f'{eval_dataset_name} : {outputs_metrics}')
        return outputs_metrics
    
    def evaluate_at_word_level(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", dataset_name=''):
        """
        Evaluate model at word level on the evaluation dataset.
        
        Similar to evaluate(), but aggregates token-level predictions to the word level
        before computing metrics. This provides a more meaningful evaluation for emphasis
        detection since emphasis typically applies to entire words, not individual tokens.
        
        A word is considered emphasized if any of its tokens are predicted as emphasized.
        
        Args:
            eval_dataset: Dataset to evaluate on
            ignore_keys: Keys to ignore in the model output
            metric_key_prefix: Prefix for metric keys in output
            dataset_name: Name of the dataset for logging purposes
            
        Returns:
            Dictionary of word-level evaluation metrics
        """
        # Word-level evaluation: aggregate token predictions to words, then compute metrics.
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()

        all_preds_by_words = []
        all_labels_by_words = []

        for batch in tqdm(eval_dataloader):
            # Extract input features and labels
            input_features = batch["input_features"]
            labels_keys = [elem for i, elem in enumerate(batch.keys()) if "labels" in elem and not "labels_head" in elem][-1]
            whisper_labels = batch[labels_keys]
            labels_head_keys = [elem for i, elem in enumerate(batch.keys()) if "labels_head" in elem][-1]
            labels_head = batch[labels_head_keys]

            # Generate predictions by a forward pass through the model
            with torch.no_grad():
                # generated_ids = self.model.generate(
                #     input_features=input_features,
                #     # decoder_input_ids=batch.get("decoder_input_ids"),  # If required
                #     max_length=self.args.generation_max_length,
                #     # labels_head=None,
                #     whisper_labels=whisper_labels,
                #     # **self.args.generation_kwargs
                # )
                generated_ids = self.model(
                    input_features=input_features,
                    labels_head=labels_head,
                    whisper_labels=whisper_labels,
                )['preds']
                
            all_labels_head_by_words = []
            all_generated_ids_by_words = []
            batch_samples = torch.where(torch.tensor(eval_dataset['sentence_index']).cpu() == batch['sentence_index'].unsqueeze(1).cpu())[1].numpy()
            map_dict_key = [elem for i, elem in enumerate(eval_dataset.column_names) if "map_dict" in elem][-1]
            for i in range(labels_head.shape[0]):
                j_start = 1
                labels_head_by_words = [-100]
                generated_ids_by_words = [-100]
                for val in eval_dataset[int(batch_samples[i])][map_dict_key]["values"]:
                    j_end = j_start + (len(val) if len(val) > 0 else 1)
                    while (
                        j_end < whisper_labels.shape[1] and
                        not np.array_equal(
                            whisper_labels[i][j_start:j_end].cpu().numpy(),
                            np.array(val)
                        )
                        and not whisper_labels[i][j_end].item() == 50256
                        and not len(val) == 0
                    ):
                        if labels_head[i][j_start].item() == 1:
                            labels_head_by_words.append(1)
                        else:
                            labels_head_by_words.append(0)
                        if generated_ids[i][j_start].item() == 1:
                            generated_ids_by_words.append(1)
                        else:
                            generated_ids_by_words.append(0)
                        j_start += 1
                        j_end += 1
                    if (labels_head[i][j_start:j_end] == 1).any().item():
                        labels_head_by_words.append(1)
                    else:
                        labels_head_by_words.append(0)
                    if (generated_ids[i][j_start:j_end] == 1).any().item():
                        generated_ids_by_words.append(1)
                    else:
                        generated_ids_by_words.append(0)
                    j_start = j_end
                # add the last punctuation mark if it's not the end of the sequence
                if j_end < whisper_labels.shape[1] and whisper_labels[i][j_end].item() != 50256:
                    if labels_head[i][j_end].item() == 1:
                        labels_head_by_words.append(1)
                    else:
                        labels_head_by_words.append(0)
                    if generated_ids[i][j_end].item() == 1:
                        generated_ids_by_words.append(1)
                    else:
                        generated_ids_by_words.append(0)
                    j_end += 1
                if j_end < labels_head.shape[1]:
                    assert labels_head[i][j_end]==-100
                    
                labels_head_by_words_padded = self._pad_tensors_to_max_len(
                    torch.tensor(labels_head_by_words).unsqueeze(0), max_length=self.args.generation_max_length
                )
                generated_ids_by_words_padded = self._pad_tensors_to_max_len(
                    torch.tensor(generated_ids_by_words).unsqueeze(0), max_length=self.args.generation_max_length
                )                
                all_generated_ids_by_words.append(generated_ids_by_words_padded.squeeze(0))
                all_labels_head_by_words.append(labels_head_by_words_padded.squeeze(0))
                
            padded_labels_by_words = torch.stack(all_labels_head_by_words)
            padded_preds_by_words = torch.stack(all_generated_ids_by_words)
            
            all_preds_by_words.append(padded_preds_by_words.cpu().numpy())
            all_labels_by_words.append(padded_labels_by_words.cpu().numpy())

        # Flatten lists        
        all_preds_by_words = np.concatenate(all_preds_by_words, axis=0)
        all_labels_by_words = np.concatenate(all_labels_by_words, axis=0)

        # Compute metrics
        outputs_metrics = {}
        if self.compute_metrics is not None:
            metrics = self.compute_metrics(
                {"predictions": all_preds_by_words, "label_ids": all_labels_by_words}
            )
            for key, value in metrics.items():
                key = f"{metric_key_prefix}_{key}"
                if isinstance(value, np.ndarray):
                    outputs_metrics[key] = value.tolist()
                else:
                    outputs_metrics[key] = value
                    
        with open(os.path.join(self.args.output_dir, "log_eval_word_level.txt"), "a") as file:
            json.dump(f'Evaluate at WORD LEVEL {dataset_name}:', file)
            json.dump(outputs_metrics, file)
        self.log(outputs_metrics)
        return outputs_metrics

    def align_samples_aux(self, pred):
        """
        Identify samples where predictions and labels have mismatched lengths.
        
        Used to filter out problematic samples where the model's predictions
        cannot be directly compared to ground truth labels due to length mismatch.
        
        Args:
            pred: Dictionary containing 'predictions' and 'label_ids' arrays
            
        Returns:
            List of row indices to remove from evaluation
        """
        # Identify sample indices where prediction and label lengths mismatch.
        pred_ids = pred["predictions"]
        label_ids = pred["label_ids"]
        pad_token_id = -100

        rows_to_remove = []
        for i, (pred_id, label_id) in enumerate(zip(pred_ids, label_ids)):
            # Create a mask where pred_ids are not equal to pad_token_id
            mask_pred_ids = pred_id != pad_token_id
            # Create a mask where label_ids are not equal to pad_token_id
            mask_label_ids = label_id != pad_token_id
            if pred_id[mask_pred_ids].shape[0] != label_id[mask_label_ids].shape[0]:
                rows_to_remove.append(i)

        return rows_to_remove
    
    def aligned_whisper_transcriptions(self, example):
        """
        Generate Whisper transcriptions and check alignment with ground truth.
        
        Used during dataset preprocessing to identify samples where the Whisper model's
        transcription matches the ground truth transcription, ignoring formatting
        differences like capitalization and punctuation.
        
        Args:
            example: Dataset example containing audio and transcription
            
        Returns:
            Example with added 'aligned_whisper_transcriptions' field
        """
        # Generate transcription with Whisper and align it with ground truth.
        token_ids = self.model.whisper_model.generate(input_features=example['input_features'].to('cuda').unsqueeze(0), 
                                                    labels=example['whisper_labels'].to('cuda').unsqueeze(0))
        transcription = self.model.processor.tokenizer.decode(token_ids[0], skip_special_tokens=True)
        example['aligned_whisper_transcriptions'] = ''
        if transcription.lstrip().lower().replace(',','').replace('.','') == example['transcription'].lower():
            example['aligned_whisper_transcriptions'] = transcription
        return example
    
    def filter_misaligned_samples(self, example):
        """
        Filter out examples where Whisper transcription doesn't align with ground truth.
        
        Args:
            example: Dataset example containing aligned_whisper_transcriptions
            
        Returns:
            Boolean indicating whether the example should be kept (True) or filtered out (False)
        """
        # Filter samples where Whisper transcription is inconsistent with ground truth.
        return example['aligned_whisper_transcriptions'] != ''

    def align_samples(self, dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Process dataset to identify and flag misaligned samples.
        
        Generates model predictions for each example in the dataset, and identifies
        examples where the prediction length doesn't match the label length, which
        would cause evaluation errors.
        
        Args:
            dataset: Dataset to check for alignment issues
            ignore_keys: Keys to ignore in the model output
            metric_key_prefix: Prefix for metric keys in output
            
        Returns:
            List of indices for samples that should be removed due to alignment issues
        """
        # Check for samples in the dataset where prediction and label lengths mismatch.
        eval_dataloader = self.get_eval_dataloader(dataset)
        self.model.eval()

        all_preds = []
        all_labels = []

        for i, batch in enumerate(tqdm(eval_dataloader)):
            # Extract input features and labels
            input_features = batch["input_features"]
            whisper_labels = batch["whisper_labels"]
            labels_head = batch["labels_head"]

            # Generate predictions
            with torch.no_grad():
                # Adjust inputs according to your model's requirements
                generated_ids = self.model.generate(
                    input_features=input_features,
                    whisper_labels=whisper_labels,
                )

            # Pad or truncate predictions and labels to a fixed length
            padded_preds = self._pad_tensors_to_max_len(
                generated_ids, max_length=self.args.generation_max_length
            )
            padded_labels = self._pad_tensors_to_max_len(
                labels_head, max_length=self.args.generation_max_length
            )
            # Collect predictions and labels
            all_preds.append(padded_preds.cpu().numpy())
            all_labels.append(padded_labels.cpu().numpy())

        # Flatten lists
        for i in range(len(all_preds)):
            print(f"{all_preds[i].shape=}, {all_labels[i].shape=}")
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        return self.align_samples_aux(
            {"predictions": all_preds, "label_ids": all_labels}
        )
