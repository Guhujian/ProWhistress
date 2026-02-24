# 该文件实现了用于语音序列到序列任务的数据整理器，支持自动填充、标签处理和特殊损失掩码等功能
import torch.nn.functional as F
import torch
from dataclasses import dataclass
from typing import List, Union, Any, Dict

# 数据整理器：用于语音到文本（seq2seq）任务，支持填充和多头标签处理
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any  # Whisper处理器，包含特征提取器和分词器
    decoder_start_token_id: int  # 解码器起始token id
    forced_decoder_ids: int      # 强制解码token id
    eos_token_id: int           # 句子结束token id
    transcription_column_name: str  # 转录文本列名

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
                      # features是一个字典列表，每个字典包含音频特征、转录标签、重音标签等
        Returns:
            Dictionary containing padded tensors ready for model input:
            - input_features: Padded audio features
            - whisper_labels: Padded token IDs for transcription (with -100 for padding)
            - labels_head: Padded binary vectors for emphasis detection (with -100 for padding)
            - sentence_index: Tensor of sentence indices

        
        """
        # 步骤1：提取并填充音频输入特征
        # 找到包含input_features的key（兼容不同命名）
        input_features_key = [elem for elem in list(features[0].keys()) if "input_features" in elem][0]
        input_features = [
            {"input_features": feature[input_features_key]} for feature in features
        ]
        # 使用Whisper的特征提取器进行填充，返回张量
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        
        # 步骤2：确定Whisper转录标签的正确列名
        whisper_labels_key = 'whisper_labels'
        # 兼容不同数据集，自动查找包含labels且不是head的列名
        whisper_labels_key_opts = [elem for elem in list(features[0].keys()) if "labels" in elem and not "head" in elem and self.transcription_column_name in elem]
        if whisper_labels_key_opts != []:
            whisper_labels_key = whisper_labels_key_opts[0]
        if len(whisper_labels_key_opts) > 1:
            raise ValueError(
                f"More than one whisper_labels (backbone model labels) candidate found in features: {whisper_labels_key_opts}"
            )
            
        # 步骤3：提取并填充转录标签序列
        labels = [
            {"input_ids": feature[whisper_labels_key]} for feature in features
        ]
        # 用tokenizer进行填充，返回张量
        labels = self.processor.tokenizer.pad(labels, return_tensors="pt")

        # 用-100替换padding部分，便于loss忽略
        labels = labels["input_ids"].masked_fill(
            labels.attention_mask.ne(1), -100
        )

        # 步骤4：确定重音检测head标签的列名
        labels_head_key = 'labels_head'
        labels_head_key_opts = [elem for elem in list(features[0].keys()) if "labels_head" in elem and self.transcription_column_name in elem]
        if labels_head_key_opts != []:
            labels_head_key = labels_head_key_opts[0]
        if len(labels_head_key_opts) > 1:
            raise ValueError(
                f"More than one labels_head (added decoder head labels) candidate found in features: {labels_head_key_opts}"
            )
        # 处理labels_head（自定义head标签）
        labels_head = [
            {"labels_head": feature[labels_head_key]} for feature in features
        ]
        # 计算最大长度，所有样本填充到同一长度
        max_len = max(
            [len(f["labels_head"]) for f in labels_head]
        )  # Find max length
        labels_head = torch.stack(
            [
                F.pad(
                    torch.tensor(f["labels_head"]), (0, max_len - len(f["labels_head"])), value=-100
                )
                for f in labels_head
            ]
        )

        # 步骤5：对labels_head中的特殊token进行掩码处理
        ignore_tokens = [
            self.decoder_start_token_id,
            self.forced_decoder_ids,
            self.eos_token_id,
        ]

        # 如果BOS token已在前面加过，这里去掉（后续会再加）
        labels_head = torch.where(
            torch.isin(labels, torch.tensor(list(ignore_tokens))),
            torch.tensor(-100),
            labels_head,
        )
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
            labels_head = labels_head[:, 1:]

        # 步骤6：构建最终batch字典
        batch['whisper_labels'] = labels
        batch['labels_head'] = labels_head
        batch["sentence_index"] = torch.tensor([feature["sentence_index"] for feature in features])
        # 检查labels_head和whisper_labels形状一致
        assert batch['labels_head'].shape == batch['whisper_labels'].shape

        return batch
