# data_collator.py implements a data collator for speech seq2seq tasks,
# including automatic padding, label processing, and special loss masking.
import torch.nn.functional as F
import torch
from dataclasses import dataclass
from typing import List, Union, Any, Dict

# Data collator for speech-to-text (seq2seq) tasks with padding and multi-head label handling.
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any  # Whisper processor, including feature extractor and tokenizer
    decoder_start_token_id: int  # Decoder start token id
    forced_decoder_ids: int      # Forced decoder token id
    eos_token_id: int           # End-of-sentence token id
    transcription_column_name: str  # Transcription text column name

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate and prepare batch for training or inference.
        
        This method handles:
        1. Padding audio features to the same length
        2. Identifying and preparing Whisper transcription labels
        3. Preparing emphasis detection head labels
        4. Masking special tokens for loss calculation
        5. Building a complete batch with all necessary tensors
        
        Args:
            features: List of feature dictionaries, each containing input features,
                      transcription labels, and emphasis labels
                      # features is a list of dictionaries, each containing audio
                      # features, transcription labels, emphasis labels, etc.
        Returns:
            Dictionary containing padded tensors ready for model input:
            - input_features: Padded audio features
            - whisper_labels: Padded token IDs for transcription (with -100 for padding)
            - labels_head: Padded binary vectors for emphasis detection (with -100 for padding)
            - sentence_index: Tensor of sentence indices

        
        """
        # Step 1: Extract and pad audio input features
        # Find the key containing input_features (compatible with different naming styles)
        input_features_key = [elem for elem in list(features[0].keys()) if "input_features" in elem][0]
        input_features = [
            {"input_features": feature[input_features_key]} for feature in features
        ]
        # Pad using Whisper feature extractor and return tensors
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        
        # Step 2: Determine the correct column name for Whisper transcription labels
        whisper_labels_key = 'whisper_labels'
        # For dataset compatibility, auto-detect label columns containing "labels" but not "head"
        whisper_labels_key_opts = [elem for elem in list(features[0].keys()) if "labels" in elem and not "head" in elem and self.transcription_column_name in elem]
        if whisper_labels_key_opts != []:
            whisper_labels_key = whisper_labels_key_opts[0]
        if len(whisper_labels_key_opts) > 1:
            raise ValueError(
                f"More than one whisper_labels (backbone model labels) candidate found in features: {whisper_labels_key_opts}"
            )
            
        # Step 3: Extract and pad transcription label sequences
        labels = [
            {"input_ids": feature[whisper_labels_key]} for feature in features
        ]
        # Pad with tokenizer and return tensors
        labels = self.processor.tokenizer.pad(labels, return_tensors="pt")

        # Replace padding positions with -100 so they are ignored in loss
        labels = labels["input_ids"].masked_fill(
            labels.attention_mask.ne(1), -100
        )

        # Step 4: Determine the column name for emphasis-head labels
        labels_head_key = 'labels_head'
        labels_head_key_opts = [elem for elem in list(features[0].keys()) if "labels_head" in elem and self.transcription_column_name in elem]
        if labels_head_key_opts != []:
            labels_head_key = labels_head_key_opts[0]
        if len(labels_head_key_opts) > 1:
            raise ValueError(
                f"More than one labels_head (added decoder head labels) candidate found in features: {labels_head_key_opts}"
            )
        # Process labels_head (custom head labels)
        labels_head = [
            {"labels_head": feature[labels_head_key]} for feature in features
        ]
        # Convert labels_head to Tensor when provided as list
        for f in labels_head:
            if not isinstance(f["labels_head"], torch.Tensor):
                f["labels_head"] = torch.tensor(f["labels_head"], dtype=torch.long)
        # Compute max length and pad all samples to the same length
        max_len = max(
            [len(f["labels_head"]) for f in labels_head]
        )  # Find max length
        labels_head = torch.stack(
            [
                F.pad(
                    f["labels_head"], (0, max_len - f["labels_head"].shape[0]), value=-100
                )
                for f in labels_head
            ]
        )

        # Step 5: Mask special tokens in labels_head
        ignore_tokens = [
            self.decoder_start_token_id,
            self.forced_decoder_ids,
            self.eos_token_id,
        ]

        # If BOS token has already been prepended, remove it here (it will be added later)
        labels_head = torch.where(
            torch.isin(labels, torch.tensor(list(ignore_tokens))),
            torch.tensor(-100),
            labels_head,
        )
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
            labels_head = labels_head[:, 1:]

        # Step 6: Build the final batch dictionary
        batch['whisper_labels'] = labels
        batch['labels_head'] = labels_head
        batch["sentence_index"] = torch.tensor([feature["sentence_index"] for feature in features])
        # Ensure labels_head and whisper_labels have matching shapes
        assert batch['labels_head'].shape == batch['whisper_labels'].shape

        return batch
