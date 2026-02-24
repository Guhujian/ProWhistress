print("🚀 脚本已启动，正在加载依赖包...") # 加在文件最第一行
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import json
from tqdm import tqdm
from transformers import WhisperConfig
# 引入你的项目模块 (根据实际目录结构调整引用)
from ..model.model import WhiStress
from .data_loader import load_data

def analyze_gate_on_test_set(args):
    # 1. 设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. 加载模型配置与权重
    print(f"Loading model from {args.model_path}...")
    
    # 读取元数据以恢复配置
    with open(os.path.join(args.model_path, "metadata.json"), "r") as f:
        meta = json.load(f)
    
    # 初始化模型结构 (必须与训练时一致)
    config = WhisperConfig()
    model = WhiStress(
        config,
        d_ctx=meta.get("d_ctx", 256),
        stress_encoder_layers=meta.get("stress_encoder_layers", 2),
        stress_encoder_input_layer=meta.get("stress_encoder_input_layer", 12),
        decoder_input_layer=meta.get("decoder_input_layer", 12),
        whisper_backbone_name=args.whisper_path
    )
    
    # 加载权重
    model.load_model(args.model_path, device=device)
    model.to(device)
    model.eval()

    # 3. 加载测试集
    print(f"Loading test dataset: {args.dataset_eval}...")
    dataset_obj = load_data(model, args.transcription_column_name, dataset_name=args.dataset_eval, save_path=args.dataset_path)
    _, _, test_dataset = dataset_obj.split_train_val_test()
    
    # 随机采样几个样本，或者指定索引
    # 这里我们简单地取前 5 个样本进行分析
    num_samples_to_plot = 20
    indices = range(num_samples_to_plot)

    # 4. 注册 Hook：抓取 Gate 值
    gate_activations = {} # 用于存储当前 forward 的 gate 值
    
    def get_gate_hook(module, input, output):
        # output shape: [Batch, Seq_Len, Dim]
        # 我们取平均值，变成 [Batch, Seq_Len]，代表该 Token 在所有特征维度上的平均门控强度
        gate_mean = output.mean(dim=-1).detach().cpu()
        gate_activations['current'] = gate_mean

    # 注册到 Sigmoid 层 (gate_net 的最后一层)
    hook_handle = model.fusion_gate.gate_net[-1].register_forward_hook(get_gate_hook)

    # 5. 循环推理并绘图
    print(f"Analyzing first {num_samples_to_plot} samples...")
    os.makedirs(args.output_dir, exist_ok=True)

    for i in indices:
        sample = test_dataset[i]
        
        # 准备输入
        input_features = torch.tensor(sample["input_features"]).unsqueeze(0).to(device) # [1, 80, 3000]
        whisper_labels = torch.tensor(sample[f"labels_{args.transcription_column_name}"]).unsqueeze(0).to(device) # [1, Seq_Len]
        labels_head = torch.tensor(sample[f"labels_head_{args.transcription_column_name}"]).unsqueeze(0).to(device) # [1, Seq_Len]

        # 为了让 forward 跑通，我们需要构造 decoder_input_ids
        # Whisper 的 labels 通常是 -100 填充的，我们需要把它转回 input_ids (去除 -100)
        decoder_input_ids = whisper_labels.clone()
        decoder_input_ids[decoder_input_ids == -100] = model.config.pad_token_id
        # Shift right logic (Whisper通常需要start token，这里简化处理，直接用labels作为forcing)
        # 严格来说应该 shift right，但在 forward 中如果传入 labels，模型内部会自动处理 shift
        
        with torch.no_grad():
            # 前向传播
            outputs = model(
                input_features=input_features,
                whisper_labels=whisper_labels, # 传入labels让模型内部计算decoder_input_ids
                labels_head=labels_head
            )
            
            # 获取预测结果 (0或1)
            preds = outputs.preds.cpu().numpy()[0] # [Seq_Len]
            
            # 获取 Gate 值
            gate_vals = gate_activations['current'][0].numpy() # [Seq_Len]

        # --- 数据后处理与可视化 ---
        
        # 1. 解码文本 (把 Token ID 变成单词)
        # 过滤掉 -100 的标签
        valid_mask = whisper_labels[0].cpu() != -100
        valid_tokens = whisper_labels[0][valid_mask].cpu().tolist()
        text_tokens = [model.processor.tokenizer.decode([t]) for t in valid_tokens]
        
        # 对齐长度：Gate 值、预测值、真实标签都需要截取有效部分
        # 注意：Forward pass 输出的长度可能和 label 长度一致
        seq_len = len(text_tokens)
        current_gate = gate_vals[:seq_len]
        current_preds = preds[:seq_len]
        current_labels = labels_head[0][valid_mask].cpu().numpy()

        # 过滤特殊 tokens
        ignore_tokens = {"<|startoftranscript|>", "<|endoftext|>", "<|notimestamps|>"}
        keep_indices = [i for i, t in enumerate(text_tokens) if t not in ignore_tokens]
        
        text_tokens = [text_tokens[i] for i in keep_indices]
        current_gate = current_gate[keep_indices]
        current_preds = current_preds[keep_indices]
        current_labels = current_labels[keep_indices]
        seq_len = len(text_tokens)

        # 保存绘图数据，方便后续合并绘图
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

        # 2. 绘图
        plt.figure(figsize=(15, 6))
        
        # 绘制 Gate 值曲线
        x = range(seq_len)
        plt.plot(x, current_gate, label='Gate Activation (Avg)', color='blue', linewidth=2, marker='o', markersize=4)
        
        # 标记真实重音位置 (Ground Truth) - 用红色背景条表示
        for idx, label in enumerate(current_labels):
            if label == 1:
                plt.axvspan(idx - 0.4, idx + 0.4, color='red', alpha=0.3, label='GT Stress' if 'GT Stress' not in plt.gca().get_legend_handles_labels()[1] else "")

        # 标记预测重音位置 - 用绿色星星表示
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