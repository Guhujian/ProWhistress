#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文版本WhiStress模型训练脚本
基于英文版本适配中文语音重音检测任务
使用Aishell_final_with_char_mapping数据集
"""

import torch
import numpy as np
import os
import sys
import logging
import argparse
from pathlib import Path
from transformers import WhisperConfig, Seq2SeqTrainingArguments, TrainerCallback, EarlyStoppingCallback, set_seed

# 导入模型和训练相关模块
from whistress.model.model import WhiStress
from whistress.training.data_loader import load_data
from whistress.training.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from whistress.training.trainer import WhiStressTrainer
from whistress.training.metrics import WhiStressMetrics

CURRENT_DIR = Path(__file__).parent
WANDB_API_KEY = os.environ.get("WAND_API_KEY", None)

# 自定义回调，支持保存时处理特殊类型并输出最佳模型提示
class CustomCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        # 使用根日志记录器确保输出到终端
        self.logger = logging.getLogger("whistress.training.train")
        # 确保日志级别足够低以显示INFO消息
        self.logger.setLevel(logging.INFO)
    
    def on_save(self, args, state, control, **kwargs):
        # Example of handling serialization
        if hasattr(state, "metrics") and isinstance(state.metrics, np.ndarray):
            state.metrics = state.metrics.tolist()
    
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """在每次评估后输出指标，最佳模型判断与保存交由Transformers内置流程处理"""
        # 添加调试信息
        print(f"🔍 CustomCallback.on_evaluate 被调用 - 步骤: {state.global_step}")
        sys.stdout.flush()  # 强制刷新输出缓冲区
        
        # 兼容 metrics 参数（Transformers 标准传递的是 metrics 而不是 logs）
        if logs is None and "metrics" in kwargs:
            logs = kwargs["metrics"]
        
        if logs is None:
            print("⚠️ logs 为 None，跳过评估")
            sys.stdout.flush()
            return
            
        # 获取当前的评估指标
        current_metric_key = args.metric_for_best_model
        print(f"🎯 寻找指标: {current_metric_key}")
        print(f"📋 可用指标: {list(logs.keys())}")
        sys.stdout.flush()
        
        if current_metric_key not in logs:
            print(f"❌ 指标 {current_metric_key} 不在日志中")
            sys.stdout.flush()
            return
            
        current_metric = logs[current_metric_key]
        print(f"📊 当前 {current_metric_key}: {current_metric:.4f}")
        
        # 使用Transformers内置的最优模型跟踪机制
        # 检查state.best_metric是否存在且已更新
        state_best_metric = getattr(state, 'best_metric', None)
        print(f"🏆 Transformers内置最优指标: {state_best_metric}")
        sys.stdout.flush()
        
        # 检查是否是新的最优模型 - 使用Transformers的逻辑
        is_better = False
        if state_best_metric is None:
            is_better = True
            print(f"🆕 首次评估，设为最优模型")
        else:
            # 比较当前指标与Transformers跟踪的最优指标
            if args.greater_is_better:
                is_better = current_metric > state_best_metric
            else:
                is_better = current_metric < state_best_metric
            print(f"🔄 比较: 当前={current_metric:.4f} vs Transformers最优={state_best_metric:.4f}, 更好={is_better}")
        sys.stdout.flush()
        
        if is_better:
            print(f"🎉 发现新的最优模型! {current_metric_key}: {current_metric:.4f}")
            print(f"📊 当前评估指标 - Accuracy: {logs.get('eval_accuracy', 'N/A'):.4f}, "
                           f"Precision: {logs.get('eval_precision', 'N/A'):.4f}, "
                           f"Recall: {logs.get('eval_recall', 'N/A'):.4f}, "
                           f"F1: {logs.get('eval_f1', 'N/A'):.4f}")
            sys.stdout.flush()
            
            self.logger.info(f"🎉 发现新的最优模型! {current_metric_key}: {current_metric:.4f}")
            self.logger.info(f"📊 当前评估指标 - Accuracy: {logs.get('eval_accuracy', 'N/A'):.4f}, "
                           f"Precision: {logs.get('eval_precision', 'N/A'):.4f}, "
                           f"Recall: {logs.get('eval_recall', 'N/A'):.4f}, "
                           f"F1: {logs.get('eval_f1', 'N/A'):.4f}")
        else:
            print(f"📈 当前模型未超越最优模型 (Transformers最优: {state_best_metric:.4f})")
            sys.stdout.flush()


def train_or_evaluate(args):
    """主训练或评估函数"""
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 配置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # 数据加载
    print(f"正在加载中文数据集: {args.dataset_path}")
    
    # 定义要移除的列（根据中文数据集结构）
    # 注意：sentence_index和map_dict需要保留，因为数据整理器和评估方法需要使用它们
    columns_to_remove = ['audio', 'ssml', 'voice_name', 'gender', 'timestamps_mfa']
    
    # 使用新的load_data函数
    train_dataset, eval_dataset = load_data(
        dataset_path=args.dataset_path,
        transcription_column_name=args.transcription_column_name,
        emphasis_indices_column_name='emphasis_indices',  # 中文数据集使用emphasis_indices
        columns_to_remove=columns_to_remove,
        split_train_val_percentage=0.0 if not args.is_train else 0.02 # 评估模式下不分割验证集
    )
    
    print(f"📊 训练集大小: {len(train_dataset)}")
    print(f"📊 验证集大小: {len(eval_dataset)}")
    
    # 初始化模型
    print("🤖 正在初始化WhiStress中文模型...")
    
    # 使用本地whisper模型路径
    local_whisper_path = str(CURRENT_DIR / "whistress" / "whisper_model" / "whisper" / "whisper-small")
    print(f"   - 使用本地Whisper模型: {local_whisper_path}")
    
    # 初始化Whisper配置
    whisper_config = WhisperConfig.from_pretrained(local_whisper_path)
    
    model = WhiStress(
        config=whisper_config,  # 添加必需的config参数
        layer_for_head=args.layer_for_head,  # 使用命令行参数
        d_ctx=args.d_ctx,
        stress_encoder_layers=args.stress_encoder_layers,
        whisper_backbone_name=local_whisper_path,  # 使用本地模型路径
        stress_reg_coeff=args.stress_reg_coeff,
        dropout=args.dropout,
        freeze_stress_encoder=args.freeze_stress_encoder,
        stress_encoder_input_layer=args.stress_encoder_input_layer,
        decoder_input_layer=args.decoder_input_layer,
    )
    
    # 数据整理器
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=model.processor,
        decoder_start_token_id=model.whisper_model.config.decoder_start_token_id,
        forced_decoder_ids=model.whisper_model.config.forced_decoder_ids[0][1],
        eos_token_id=model.whisper_model.config.eos_token_id,
        transcription_column_name=args.transcription_column_name
    )
    
    # 训练参数 - 针对大模型优化内存使用
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_path,
        # 大幅减少批次大小以适应stress_encoder_layers=12的内存需求
        per_device_train_batch_size=args.train_batch_size,  # 从32减少到4以节省内存
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # 增加梯度累积步数以保持有效批次大小=32
        learning_rate=5e-4, # decrease to ~5e-4 for large dataset, increase to ~4e-4 for small dataset
        warmup_ratio=0.05,
        num_train_epochs=2, # Increase to 4 for task-specific evaluation, use 2 for zero-shot - More generalized evaluation. 
        seed=args.seed,
        # gradient_checkpointing=True,  # WhiStress模型不支持梯度检查点
        # fp16=True,  # 启用半精度浮点数训练以节省显存和加速训练
        do_train=True,
        do_eval=True, 
        eval_strategy="steps",
        save_strategy="steps",
        per_device_eval_batch_size=32,  # 评估时也使用小批次
        generation_max_length=96,
        save_steps=100,
        save_total_limit=2,  # 只保存最新的2个检查点
        load_best_model_at_end=True,  # 训练结束时加载最优模型
        metric_for_best_model="eval_f1",  # 以验证集F1分数作为最优模型判断标准
        greater_is_better=True,  # 对于F1分数，越大越好
        eval_steps=10,
        logging_steps=10,
        weight_decay=0.01,
        dataloader_num_workers=8,
        push_to_hub=False,
        report_to=['wandb'] if WANDB_API_KEY is not None else None,
        label_names=[f"labels_head_{args.transcription_column_name}", "sentence_index", f"labels_{args.transcription_column_name}"],
        overwrite_output_dir=True, # change if you want to keep previous output
    )
    # 设置随机种子
    set_seed(training_args.seed)
    metrics = WhiStressMetrics()

    # 输出训练配置
    if args.is_train:
        print("\n" + "="*60)
        print("🔧 训练配置信息")
        print("="*60)
        print(f"📁 输出目录: {args.output_path}")
        print(f"🎯 训练数据集: {args.dataset_train}")
        print(f"📊 评估数据集: {args.dataset_eval}")
        print(f"📝 转录列名: {args.transcription_column_name}")
        print(f"🧠 模型配置:")
        print(f"   - Whisper 模型: {local_whisper_path}")
        print(f"   - StressEncoder 输入层 (Whisper Encoder): {args.stress_encoder_input_layer}")
        print(f"   - Additional Decoder 输入层 (Whisper Encoder): {args.decoder_input_layer}")
        print(f"   - Head 输入层 (Whisper Decoder): {args.layer_for_head}")
        print(f"   - 解码器上下文长度: {args.d_ctx}")
        print(f"   - 应力编码器层数: {args.stress_encoder_layers}")
        
        # 计算模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"📊 模型参数量:")
        print(f"   - 总参数量: {total_params:,}")
        print(f"   - 可训练参数: {trainable_params:,}")
        print(f"   - 冻结参数: {frozen_params:,}")
        print(f"   - 可训练比例: {trainable_params/total_params*100:.2f}%")
        print(f"⚙️ 训练参数:")
        print(f"   - 训练批次大小: {training_args.per_device_train_batch_size}")
        print(f"   - 评估批次大小: {training_args.per_device_eval_batch_size}")
        print(f"   - 学习率: {training_args.learning_rate}")
        print(f"   - 训练轮数: {training_args.num_train_epochs}")
        print(f"   - 随机种子: {training_args.seed}")
        print(f"   - 梯度累积步数: {training_args.gradient_accumulation_steps}")
        print(f"   - 权重衰减: {training_args.weight_decay}")
        print(f"   - 预热比例: {training_args.warmup_ratio}")
        print(f"   - 保存步数: {training_args.save_steps}")
        print(f"   - 评估步数: {training_args.eval_steps}")
        print(f"   - 最大生成长度: {training_args.generation_max_length}")
        print(f"🎲 数据集信息:")
        if train_dataset is not None:
            print(f"   - 训练集大小: {len(train_dataset)}")
        if eval_dataset is not None:
            print(f"   - 验证集大小: {len(eval_dataset)}")
        print("="*60)
        print()
    
    # 初始化训练器
    trainer = WhiStressTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset if args.is_train else None,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=metrics.compute_metrics,
        compute_loss_func=model.loss_fct,
        processing_class=model.processor.feature_extractor,
        callbacks=[CustomCallback(), EarlyStoppingCallback(early_stopping_patience=20)],  # 添加自定义回调来监控最优模型
    )
    
    if args.is_train:
        print("🚀 开始训练中文WhiStress模型...")
        
        resume_from_checkpoint = args.resume_from_checkpoint
        if resume_from_checkpoint is not None:
             if resume_from_checkpoint.lower() == "true":
                 resume_from_checkpoint = True
             print(f"🔄 尝试从检查点恢复训练: {resume_from_checkpoint}")
        else:
             print("🆕 开始新的训练 (未指定检查点)")
        
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # 保存最终模型
        print("💾 正在保存最终模型...")
        # 将最佳模型保存到 best_model 子文件夹，以便清晰区分
        final_output_path = os.path.join(args.output_path, "best_model")
        os.makedirs(final_output_path, exist_ok=True)
        trainer.save_final_model(final_output_path, training_args)
        print(f"✅ 最佳模型已保存到: {final_output_path}")
        
        # 训练结束后进行完整评估
        print("📊 开始最终评估...")
        
        # Token级别评估
        print("🔤 进行Token级别评估...")
        eval_results_token = trainer.evaluate(
            ignore_keys=["whisper_logits"],
            eval_dataset=eval_dataset,
            dataset_name=f"{args.dataset_eval}-final-token_level"
        )
        print("📈 Token级别评估结果:")
        for key, value in eval_results_token.items():
            print(f"  {key}: {value}")
        
        # 字符级别评估
        print("🔤 进行字符级别评估...")
        eval_results_char = trainer.evaluate_at_char_level(
            ignore_keys=["whisper_logits"],
            eval_dataset=eval_dataset,
            dataset_name=f"{args.dataset_eval}-final-char_level"
        )
        print("📈 字符级别评估结果:")
        for key, value in eval_results_char.items():
            print(f"  {key}: {value}")
            
    else:
        print("📊 开始评估模型...")
        # 加载模型进行评估
        if args.model_path:
            model.load_model(args.model_path)
        
        # Token级别评估
        print("🔤 进行Token级别评估...")
        eval_results_token = trainer.evaluate(
            ignore_keys=["whisper_logits"],
            eval_dataset=eval_dataset,
            dataset_name=f"{args.dataset_eval}-eval-token_level"
        )
        print("📈 Token级别评估结果:")
        for key, value in eval_results_token.items():
            print(f"  {key}: {value}")
        
        # 字符级别评估
        print("🔤 进行字符级别评估...")
        eval_results_char = trainer.evaluate_at_char_level(
            ignore_keys=["whisper_logits"],
            eval_dataset=eval_dataset,
            dataset_name=f"{args.dataset_eval}-eval-char_level"
        )
        print("📈 字符级别评估结果:")
        for key, value in eval_results_char.items():
            print(f"  {key}: {value}")


def str2bool(v):
    """字符串转布尔值"""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="中文WhiStress模型训练脚本")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="用于评估的模型路径。如果是训练模式，不应提供此参数",
    )
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="中文数据集路径",
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="./chinese_whistress_output/encoders_3",
        help="训练输出保存路径",
    )
    
    parser.add_argument(
        "--transcription_column_name",
        type=str,
        choices=["transcription", "aligned_whisper_transcriptions"],
        default="transcription",
        help="数据集中转录文本的列名",
    )
    
    parser.add_argument(
        "--d_ctx",
        type=int,
        default=256,
        help="音频-文本交叉注意力的上下文瓶颈维度",
    )
    
    parser.add_argument(
        "--stress_encoder_layers",
        type=int,
        default=9,  # 使用与英文版本一致的层数
        help="StressEncoder中Transformer编码器层的数量",
    )
    
    parser.add_argument(
        "--layer_for_head",
        type=int,
        default=9,  # 使用与英文版本一致的层数
        help="用于应力检测头部的Whisper编码器层索引",
    )
    
    parser.add_argument(
        "--stress_encoder_input_layer",
        type=int,
        default=12,
        help="Whisper Encoder layer index to use as input for StressEncoder (default: 12, i.e., last layer)",
    )

    parser.add_argument(
        "--decoder_input_layer",
        type=int,
        default=12,
        help="Whisper Encoder layer index to use as input for Additional Decoder cross-attention (default: 12, i.e., last layer)",
    )
    
    parser.add_argument(
        "--dataset_train",
        type=str,
        choices=["Aishell_final_with_char_mapping", "Aishell_final", "SinoReal_TestOnly"],
        default="Aishell_final_with_char_mapping",
        help="用于训练和验证的数据集名称",
    )
    
    parser.add_argument(
        "--dataset_eval",
        type=str,
        choices=["Aishell_final_with_char_mapping", "Aishell_final", "SinoReal_TestOnly"],
        default="Aishell_final_with_char_mapping",
        help="用于评估的数据集名称",
    )
    
    parser.add_argument(
        "--is_train",
        type=str2bool,
        default=True,
        help="是否训练模型 (True) 或仅评估 (False)",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="The batch size per device for training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
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
        "--freeze_stress_encoder",
        type=str2bool,
        default=False,
        help="Whether to freeze the StressEncoder during training (Linear Probing).",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If specified, resumes training from this checkpoint path. Can be a path or 'True' to resume from the latest checkpoint in output_dir.",
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)
    
    # 验证参数
    if args.is_train:
        assert args.model_path is None, "训练模式下不应提供model_path参数"
    else:
        assert args.model_path is not None, "评估模式下必须提供model_path参数"
    
    # 验证数据集路径
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"数据集路径不存在: {args.dataset_path}")
    
    print("🎯 中文WhiStress模型训练配置:")
    print(f"  📁 数据集路径: {args.dataset_path}")
    print(f"  📁 输出路径: {args.output_path}")
    print(f"  🔧 应力检测头部层索引: {args.layer_for_head}")
    print(f"  🔧 上下文维度: {args.d_ctx}")
    print(f"  🔧 编码器层数: {args.stress_encoder_layers}")
    print(f"  🔧 训练批次大小: {args.train_batch_size}")
    print(f"  🔧 梯度累积步数: {args.gradient_accumulation_steps}")
    print(f"  🔧 正则化系数: {args.stress_reg_coeff}")
    print(f"  🔧 Dropout: {args.dropout}")
    print(f"  🔧 冻结StressEncoder: {args.freeze_stress_encoder}")
    print(f"  🎯 训练模式: {args.is_train}")
    print("-" * 60)
    
    train_or_evaluate(args)