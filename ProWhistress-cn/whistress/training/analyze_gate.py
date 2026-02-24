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

# 配置 Matplotlib 中文字体支持，解决乱码问题
# 优先尝试常见的中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'PingFang SC', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

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
    # 使用新的 load_data 函数接口
    _, test_dataset = load_data(
        dataset_path=args.dataset_path,
        dataset_eval=args.dataset_eval,
        transcription_column_name=args.transcription_column_name
    )
    
    if test_dataset is None:
        raise ValueError("Failed to load test dataset. Please check dataset_path and dataset_eval arguments.")
    
    # 随机采样几个样本，或者指定索引
    # 这里我们简单地取前 5 个样本进行分析
    num_samples_to_plot = 10
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
        
        # 动态查找 labels 列名
        labels_key = [k for k in sample.keys() if "labels" in k and "labels_head" not in k][0]
        labels_head_key = [k for k in sample.keys() if "labels_head" in k][0]
        
        whisper_labels = torch.tensor(sample[labels_key]).unsqueeze(0).to(device) # [1, Seq_Len]
        labels_head = torch.tensor(sample[labels_head_key]).unsqueeze(0).to(device) # [1, Seq_Len]

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
        
        # 获取原始的 token 级数据
        seq_len_tokens = len(valid_tokens)
        raw_gate_vals = gate_vals[:seq_len_tokens]
        raw_labels = labels_head[0][valid_mask].cpu().numpy()
        raw_preds = preds[:seq_len_tokens]

        # --- 聚合 Token 到 Character (字级) ---
        # 解决 Token 碎片化导致的乱码问题，并将 Gate 值聚合到字级别
        aligned_chars = []
        aligned_gates = []
        aligned_labels = []
        aligned_preds = []
        
        buffer_tokens = []
        buffer_gates = []
        buffer_labels = []
        buffer_preds = []
        
        for t, g, l, p in zip(valid_tokens, raw_gate_vals, raw_labels, raw_preds):
            buffer_tokens.append(t)
            buffer_gates.append(g)
            buffer_labels.append(l)
            buffer_preds.append(p)
            
            # 尝试解码当前 buffer
            text = model.processor.tokenizer.decode(buffer_tokens)
            
            # 如果解码结果不包含替换字符  (0xFFFD)，说明是完整的字符序列
            # 或者如果是特殊 token (如 <|start...|>)，也可以直接处理
            if "\ufffd" not in text:
                # 计算聚合指标
                # 如果多个 token 组成一个字，取平均 gate
                avg_gate = sum(buffer_gates) / len(buffer_gates)
                # Label/Pred 取最大值 (只要这部分有重音，整个字就算重音)
                max_label = max(buffer_labels)
                max_pred = max(buffer_preds)
                
                # 如果一个 token 解码出多个字 (如 "不是")，复制指标
                # 特殊处理：如果是特殊 token (如 <|startoftranscript|>)，保持整体不拆分
                if text.strip().startswith("<|") and text.strip().endswith("|>"):
                    aligned_chars.append(text)
                    aligned_gates.append(avg_gate)
                    aligned_labels.append(max_label)
                    aligned_preds.append(max_pred)
                else:
                    for char in text:
                        aligned_chars.append(char)
                        aligned_gates.append(avg_gate)
                        aligned_labels.append(max_label)
                        aligned_preds.append(max_pred)
                
                # 清空 buffer
                buffer_tokens = []
                buffer_gates = []
                buffer_labels = []
                buffer_preds = []
            else:
                # 包含乱码，可能是多字节字符的一部分，继续积累
                # 安全措施：如果 buffer 过长，强制 flush
                if len(buffer_tokens) > 5:
                    avg_gate = sum(buffer_gates) / len(buffer_gates)
                    max_label = max(buffer_labels)
                    max_pred = max(buffer_preds)
                    for char in text:
                        aligned_chars.append(char)
                        aligned_gates.append(avg_gate)
                        aligned_labels.append(max_label)
                        aligned_preds.append(max_pred)
                    buffer_tokens = []
                    buffer_gates = []
                    buffer_labels = []
                    buffer_preds = []

        # 处理剩余 buffer
        if buffer_tokens:
            text = model.processor.tokenizer.decode(buffer_tokens)
            avg_gate = sum(buffer_gates) / len(buffer_gates)
            max_label = max(buffer_labels)
            max_pred = max(buffer_preds)
            
            if text.strip().startswith("<|") and text.strip().endswith("|>"):
                aligned_chars.append(text)
                aligned_gates.append(avg_gate)
                aligned_labels.append(max_label)
                aligned_preds.append(max_pred)
            else:
                for char in text:
                    aligned_chars.append(char)
                    aligned_gates.append(avg_gate)
                    aligned_labels.append(max_label)
                    aligned_preds.append(max_pred)

        # 更新绘图用的数据
        text_tokens = aligned_chars
        current_gate = aligned_gates
        current_labels = aligned_labels
        current_preds = aligned_preds
        
        # 过滤特殊 tokens
        filtered_indices = [i for i, t in enumerate(text_tokens) if t not in ["<|startoftranscript|>", "<|endoftext|>", "<|notimestamps|>"]]
        text_tokens = [text_tokens[i] for i in filtered_indices]
        current_gate = [current_gate[i] for i in filtered_indices]
        current_labels = [current_labels[i] for i in filtered_indices]
        current_preds = [current_preds[i] for i in filtered_indices]
        
        seq_len = len(text_tokens)

        # 保存绘图数据，方便后续合并绘图
        data_to_save = {
            "tokens": text_tokens,
            "gate_values": [float(x) for x in current_gate],
            "labels": [int(x) for x in current_labels],
            "preds": [int(x) for x in current_preds]
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
    parser.add_argument("--whisper_path", type=str, default="./whistress/whisper_model/whisper/whisper-small", help="Path to local whisper model")
    args = parser.parse_args()
    
    analyze_gate_on_test_set(args)