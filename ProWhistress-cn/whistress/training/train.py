# 该文件为WhiStress模型的训练主入口，包含训练、评估、参数解析等流程
import torch
import numpy as np
import os
import sys
import logging
import argparse
from pathlib import Path
from transformers import WhisperConfig, Seq2SeqTrainingArguments, TrainerCallback, EarlyStoppingCallback, set_seed
from transformers.trainer_utils import get_last_checkpoint
from ..model.model import WhiStress
from .data_loader import load_data
from .data_collator import DataCollatorSpeechSeq2SeqWithPadding
from .trainer import WhiStressTrainer
from .metrics import WhiStressMetrics

CURRENT_DIR = Path(__file__).parent
WANDB_API_KEY = os.environ.get("WAND_API_KEY", None)

# 自定义回调，支持保存时处理特殊类型
class CustomCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        # 使用根日志记录器确保输出到终端
        self.logger = logging.getLogger("whistress.training.train")
        # 确保日志级别足够低以显示INFO消息
        self.logger.setLevel(logging.INFO)
        # 不再需要自定义的best_metric，使用Transformers内置的跟踪机制
    
    def on_save(self, args, state, control, **kwargs):
        # Example of handling serialization
        if hasattr(state, "metrics") and isinstance(state.metrics, np.ndarray):
            state.metrics = state.metrics.tolist()
    
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """在每次评估后输出指标，最佳模型判断与保存交由Transformers内置流程处理"""
        # 添加调试信息
        print(f"🔍 CustomCallback.on_evaluate 被调用 - 步骤: {state.global_step}")
        sys.stdout.flush()  # 强制刷新输出缓冲区
        
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

# 主训练/评估流程
# args: 命令行参数对象
def train_or_evaluate(args):
    # 选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    whisper_backbone_name = str(CURRENT_DIR.parent / "whisper_model" / "whisper" / "whisper-small")
    whisper_config = WhisperConfig()
    layer_for_head = 9
    
    # 打印模型配置信息
    msg = "\n" + "="*60 + "\n"
    msg += "🔧 模型配置信息\n"
    msg += "="*60 + "\n"
    msg += f"   - Whisper Encoder 提取层: 9\n"
    msg += f"   - Whisper Decoder 提取层: {layer_for_head}\n"
    msg += f"   - StressEncoder 正则化系数: {args.stress_reg_coeff}\n"
    msg += f"   - 随机种子 (Seed): {args.seed}\n"
    msg += "="*60 + "\n"
    print(msg)
    sys.stdout.flush()
    logger.info("Model Configuration:\n" + msg)

    # 加载模型或新建模型
    if args.model_path:
        logger.info(f"Loading model from {args.model_path}")
        # 尝试从保存的元数据中读取 layer_for_head 与 d_ctx
        meta_path = os.path.join(args.model_path, "metadata.json")
        d_ctx_meta = args.d_ctx
        stress_encoder_layers_meta = args.stress_encoder_layers
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    metadata = json.load(f)
                layer_for_head = metadata.get("layer_for_head", layer_for_head)
                d_ctx_meta = metadata.get("d_ctx", d_ctx_meta)
                stress_encoder_layers_meta = metadata.get("stress_encoder_layers", stress_encoder_layers_meta)
            except Exception as e:
                logger.warning(f"Failed to read metadata.json: {e}. Falling back to defaults.")
        whistress_model = WhiStress(
            whisper_config,
            layer_for_head=layer_for_head,
            whisper_backbone_name=whisper_backbone_name,
            d_ctx=d_ctx_meta,
            stress_encoder_layers=stress_encoder_layers_meta,
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
            stress_reg_coeff=args.stress_reg_coeff,
            dropout=args.dropout,
            freeze_stress_encoder=args.freeze_stress_encoder,
        ).to(device)
    
    # 设置tokenizer输入名
    whistress_model.processor.tokenizer.model_input_names = [
        "input_ids",
        "attention_mask",
        "labels_head",
    ]
    
    # 构建数据整理器
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
    # 加载训练集和验证集
    if args.is_train:
        DatasetTrain = load_data(whistress_model, args.transcription_column_name, dataset_name=args.dataset_train, save_path=args.dataset_path)
        train, val, _ = DatasetTrain.split_train_val_test()
    # 加载测试集
    DatasetEval = load_data(whistress_model, args.transcription_column_name, dataset_name=args.dataset_eval, save_path=args.dataset_path)
    _, _, test = DatasetEval.split_train_val_test()
    
    print(f"Output path for the training run: {args.output_path}")
    output_path = args.output_path

    # wandb日志
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

    # 训练参数
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_path,  # change to a repo name of your choice
        # per_device_train_batch_size=4, # assuming 8 gpus. decrease to ~2 for small dataset, increase to ~4 for large dataset.
        per_device_train_batch_size=32, # assuming 1 gpu. 
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=5e-4, # decrease to ~5e-4 for large dataset, increase to ~4e-4 for small dataset
        # warmup_ratio=0.05,
        warmup_ratio=0.05,
        num_train_epochs=2, # Increase to 4 for task-specific evaluation, use 2 for zero-shot - More generalized evaluation. 
        seed=args.seed,
        gradient_checkpointing=False,
        # fp16=True,  # 启用半精度浮点数训练以节省显存和加速训练
        do_train=True,
        do_eval=True, 
        eval_strategy="steps",
        save_strategy="steps",
        # per_device_eval_batch_size=4,
        per_device_eval_batch_size=32,
        generation_max_length=96,
        save_steps=100,
        save_total_limit=2,  # 只保存最新的2个检查点
        load_best_model_at_end=True,  # 训练结束时加载最优模型
        metric_for_best_model="eval_f1",  # 以验证集F1分数作为最优模型判断标准
        greater_is_better=True,  # 对于F1分数，越大越好
        eval_steps=10,
        logging_steps=10,
        weight_decay=0.01,
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
        print(f"   - Whisper 模型: {whisper_backbone_name}")
        print(f"   - Whisper Encoder 提取层: 最后一层 (Last)")
        print(f"   - Whisper Decoder 提取层: {layer_for_head}")
        print(f"   - 解码器上下文长度: {args.d_ctx}")
        print(f"   - 应力编码器层数: {args.stress_encoder_layers}")
        
        # 计算模型参数量
        total_params = sum(p.numel() for p in whistress_model.parameters())
        trainable_params = sum(p.numel() for p in whistress_model.parameters() if p.requires_grad)
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
        if train is not None:
            print(f"   - 训练集大小: {len(train)}")
        if val is not None:
            print(f"   - 验证集大小: {len(val)}")
        if test is not None:
            print(f"   - 测试集大小: {len(test)}")
        print("="*60)
        print()

    trainer_emphasis = None
    if args.is_train:
        # 构建训练器
        trainer_emphasis = WhiStressTrainer(
        args=training_args,
        model=whistress_model,
        train_dataset=train,
        eval_dataset=val,
        data_collator=data_collator,
        compute_metrics=metrics.compute_metrics,
        compute_loss_func=whistress_model.loss_fct,
        processing_class=whistress_model.processor.feature_extractor,
        callbacks=[CustomCallback(), EarlyStoppingCallback(early_stopping_patience=30)],  # 添加自定义回调来监控最优模型
        )
        # 训练前评估
        trainer_emphasis.evaluate_at_word_level(
            # to ignore whisper_logits in the compute_metrics function (only the custom head logits are used)
            ignore_keys=["whisper_logits"],
            eval_dataset=test,
            dataset_name=f"{args.dataset_eval}-initial-word_level",
        )
        trainer_emphasis.evaluate(
            # to ignore whisper_logits in the compute_metrics function (only the custom head logits are used)
            ignore_keys=["whisper_logits"],
            eval_dataset=test,
            dataset_name=f"{args.dataset_eval}-initial",
        )
        # 开始训练
        print("🚀 开始训练...")
        
        resume_from_checkpoint = args.resume_from_checkpoint
        if resume_from_checkpoint is not None:
             if resume_from_checkpoint.lower() == "true":
                 resume_from_checkpoint = True
             elif resume_from_checkpoint.lower() == "false":
                 resume_from_checkpoint = None
        
        # 自动检测断点
        if resume_from_checkpoint is None and os.path.isdir(args.output_path):
            last_checkpoint = get_last_checkpoint(args.output_path)
            if last_checkpoint is not None:
                print(f"🔍 自动检测到检查点: {last_checkpoint}，将从该检查点恢复训练。")
                resume_from_checkpoint = last_checkpoint
        
        trainer_emphasis.train(resume_from_checkpoint=resume_from_checkpoint)
        print("✅ 训练完成!")
        
        # 保存最终模型
        print("💾 保存最终训练模型...")
        trainer_emphasis.save_final_model(args.output_path, training_args)
        
        # 如果设置了load_best_model_at_end，也保存最优模型的副本
        if training_args.load_best_model_at_end:
            best_model_path = os.path.join(args.output_path, "best_model")
            os.makedirs(best_model_path, exist_ok=True)
            print(f"💎 保存最优模型副本到: {best_model_path}")
            trainer_emphasis.save_final_model(best_model_path, training_args)
    else:
        # 仅评估模式
        trainer_emphasis = WhiStressTrainer(
            args=training_args,
            model=whistress_model,
            train_dataset=test, # we don't really use it, but the trainer requires it
            eval_dataset=test, # we don't really use it, but the trainer requires it
            data_collator=data_collator,
            compute_metrics=metrics.compute_metrics,
            tokenizer=whistress_model.processor.feature_extractor,
        )

    # 训练后评估
    trainer_emphasis.evaluate_at_word_level(
        # to ignore whisper_logits in the compute_metrics function (only the custom head logits are used)
        ignore_keys=["whisper_logits"],
        eval_dataset=test,
        dataset_name=f"{args.dataset_eval}-final-word_level",
    )
    trainer_emphasis.evaluate(
        # to ignore whisper_logits in the compute_metrics function (only the custom head logits are used)
        ignore_keys=["whisper_logits"],
        eval_dataset=test,
        dataset_name=f"{args.dataset_eval}-final",
    )

# 字符串转bool工具

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
        "--seed",
        type=int,
        default=42,
        help="Random seed for training",
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
        default=0,
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

    # 如果未指定输出目录，则自动创建
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