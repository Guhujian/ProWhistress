# 该文件实现了WhiStress模型的主结构，包含Whisper骨干网络、重音检测自定义头、模型保存/加载、推理等功能
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    PreTrainedModel,
    WhisperConfig,
)
from transformers.models.whisper.modeling_whisper import WhisperDecoderLayer, WhisperEncoder
from transformers.modeling_outputs import BaseModelOutput
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import copy
from dataclasses import dataclass
from typing import Optional
import json

# 自定义输出结构，兼容transformers的BaseModelOutput
@dataclass
class CustomModelOutput(BaseModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    head_preds: torch.FloatTensor = None
    labels_head: Optional[torch.FloatTensor] = None
    whisper_logits: torch.FloatTensor = None
    preds: Optional[torch.Tensor] = None

# 线性分类头，用于重音检测
class LinearHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearHead, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# 两层全连接神经网络分类头
class FCNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FCNN, self).__init__()
        hidden_dim = 2 * input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 门控残差融合模块
class GatedResidualFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 门控网络：输入是两个特征的拼接，输出是一个sigmoid门控值
        self.gate_net = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        # 初始化偏置，使初始Gate值接近0 (例如 -3.0 -> sigmoid(-3.0) ≈ 0.047)
        # 这样模型在训练初期会主要依赖Main分支(预训练好的)，避免Aux分支(随机初始化)的噪声干扰
        nn.init.constant_(self.gate_net[-2].bias, -3.0)

        # 也可以添加一个LayerNorm来稳定训练
        self.norm = nn.LayerNorm(d_model)

    def forward(self, main, aux):
        # main: 主路径特征 (Additional Decoder Output)
        # aux: 辅助路径特征 (Audio Context)
        
        # 计算门控值
        combined = torch.cat([main, aux], dim=-1)
        gate = self.gate_net(combined)
        
        # 融合：Main + Gate * Aux
        # 这样如果Gate接近0，就退化为只用Main；如果Gate接近1，就充分利用Aux
        fused = main + gate * aux
        
        return self.norm(fused)

# 轻量级"StressEncoder"分支：去掉Conv层，直接处理Whisper Encoder的中间层输出
class StressEncoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_ff: int = None, num_layers: int = 2):
        super().__init__()
        dim_ff = dim_ff or d_model * 4
        # 可配置个Transformer Encoder层（batch_first）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=dim_ff, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, T, D] 来自Whisper Encoder的某一层输出
        x = self.encoder(hidden_states)
        return x  # 作为 K/V 供跨注意力与附加解码器层使用

# WhiStress主模型，继承自transformers的PreTrainedModel
class WhiStress(PreTrainedModel):

    config_class = WhisperConfig
    model_input_names = ["input_features", "labels_head", "whisper_labels"]

    def __init__(
        self,
        config: WhisperConfig,
        layer_for_head: Optional[int] = None,
        whisper_backbone_name="openai/whisper-small.en",
        d_ctx: int = 256,
        stress_encoder_layers: int = 2,
        stress_encoder_input_layer: int = 12, # 新增参数：控制StressEncoder输入来自Whisper Encoder的哪一层
        decoder_input_layer: int = 12, # 新增参数：控制Additional Decoder输入来自Whisper Encoder的哪一层
        stress_reg_coeff: float = 0.0,
        dropout: float = 0.15,  # 新增 dropout 参数，默认 0.15
        freeze_stress_encoder: bool = False,
    ):
        super().__init__(config)
        self.freeze_stress_encoder = freeze_stress_encoder
        self.stress_reg_coeff = stress_reg_coeff
        self.stress_encoder_input_layer = stress_encoder_input_layer # 保存参数
        self.decoder_input_layer = decoder_input_layer # 保存参数
        self.whisper_backbone_name = whisper_backbone_name
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            self.whisper_backbone_name,
        ).eval()
        self.processor = WhisperProcessor.from_pretrained(self.whisper_backbone_name)

        input_dim = self.whisper_model.config.d_model  # Whisper隐藏层维度
        output_dim = 2  # 重音检测二分类

        config = self.whisper_model.config
        self.d_ctx = d_ctx
        # 选择可整除的注意力头数（确保 d_ctx % num_heads == 0）
        num_heads = getattr(config, "decoder_attention_heads", 8)
        if self.d_ctx % num_heads != 0:
            for h in [8, 4, 2, 1]:
                if self.d_ctx % h == 0:
                    num_heads = h
                    break
        self.num_ctx_heads = num_heads
        # 瓶颈线性层：将 decoder/encoder 隐藏状态映射到 d_ctx，再映射回 d_model
        self.dec_to_ctx = nn.Linear(input_dim, self.d_ctx, bias=False)
        self.enc_to_ctx = nn.Linear(input_dim, self.d_ctx, bias=False)
        self.ctx_to_model = nn.Linear(self.d_ctx, input_dim, bias=False)
        # 指定用于head的层号
        self.layer_for_head = -1 if layer_for_head is None else layer_for_head
        
        # 新增一个解码器层，作为自定义head的输入
        self.additional_decoder_block = WhisperDecoderLayer(config)
        
        # 用Whisper指定层的权重初始化additional_decoder_block
        # 这样可以避免随机初始化导致的分布偏移，提高训练稳定性
        # 手动指定使用第9层进行初始化（可根据需要修改此处的数字）
        source_layer_idx = 12  # 手动指定初始化来源层，可根据需要调整
        
        # 确保源层索引在有效范围内
        num_decoder_layers = len(self.whisper_model.model.decoder.layers)
        if source_layer_idx >= num_decoder_layers:
            source_layer_idx = num_decoder_layers - 1
            
        # 深拷贝Whisper解码器指定层的权重到additional_decoder_block
        source_layer = self.whisper_model.model.decoder.layers[source_layer_idx]
        self.additional_decoder_block.load_state_dict(source_layer.state_dict())
        
        # 初始化 StressEncoder
        self.stress_encoder = StressEncoder(
            d_model=input_dim,
            num_heads=num_heads,
            num_layers=stress_encoder_layers
        )
        
        # Dropout layer for StressEncoder output to prevent overfitting
        self.stress_dropout = nn.Dropout(p=dropout)

        # 可训练的跨注意力使用 d_ctx 作为 embed_dim
        self.audio_feature_extractor = nn.MultiheadAttention(
            embed_dim=self.d_ctx,
            num_heads=self.num_ctx_heads,
            batch_first=True,
        )
        
        # 门控残差融合模块
        self.fusion_gate = GatedResidualFusion(input_dim)
        
        # 分类器输入维度现在是 input_dim (因为融合后维度不变)
        self.classifier = FCNN(input_dim, output_dim)
        # 定义带权重的交叉熵损失（正负样本不均衡）
        neg_weight = 1.0
        pos_weight = 0.7 / 0.3
        class_weights = torch.tensor([neg_weight, pos_weight])
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights)

    def to(self, device: str = ("cuda" if torch.cuda.is_available() else "cpu")):
        # 将所有子模块转移到指定设备
        self.whisper_model.to(device)
        self.additional_decoder_block.to(device)
        self.stress_encoder.to(device)
        self.dec_to_ctx.to(device)
        self.enc_to_ctx.to(device)
        self.ctx_to_model.to(device)
        self.audio_feature_extractor.to(device)
        self.fusion_gate.to(device)
        self.classifier.to(device)
        super().to(device)
        return self

    def load_model(self, save_dir=None, device="cuda" if torch.cuda.is_available() else "cpu"):
        # 只加载本地保存的自定义head、附加解码器层和音频特征提取模块参数
        if save_dir is not None:
            print('loading model from:', save_dir)
            self.classifier.load_state_dict(
                torch.load(
                    os.path.join(save_dir, "classifier.pt"),
                    weights_only=False,
                    map_location=torch.device(device),
                )
            )
                
            # 只有当保存的additional_decoder_block权重文件存在时才加载
            additional_decoder_path = os.path.join(save_dir, "additional_decoder_block.pt")
            if os.path.exists(additional_decoder_path):
                self.additional_decoder_block.load_state_dict(
                    torch.load(
                        additional_decoder_path,
                        weights_only=False,
                        map_location=torch.device(device),
                    )
                )
            else:
                print("No saved additional_decoder_block found, keeping pre-trained initialization")
            # 加载重音编码器（如果存在）
            stress_encoder_path = os.path.join(save_dir, "stress_encoder.pt")
            if os.path.exists(stress_encoder_path):
                self.stress_encoder.load_state_dict(
                    torch.load(
                        stress_encoder_path,
                        weights_only=False,
                        map_location=torch.device(device),
                    )
                )
            else:
                print("No saved stress_encoder found, keeping randomly initialized branch")
            # 加载可训练音频特征提取模块（如果存在）
            audio_path = os.path.join(save_dir, "audio_feature_extractor.pt")
            if os.path.exists(audio_path):
                sd = torch.load(
                    audio_path,
                    weights_only=False,
                    map_location=torch.device(device),
                )
                in_w = sd.get("in_proj_weight", None)
                out_w = sd.get("out_proj.weight", None)
                ok = True
                if in_w is not None and in_w.shape[1] != self.d_ctx:
                    ok = False
                if out_w is not None and out_w.shape[0] != self.d_ctx:
                    ok = False
                if ok:
                    self.audio_feature_extractor.load_state_dict(sd)
                else:
                    print(
                        f"Skip loading audio_feature_extractor: shape mismatch with d_ctx={self.d_ctx}"
                    )
            
            # 加载门控融合模块（如果存在）
            fusion_gate_path = os.path.join(save_dir, "fusion_gate.pt")
            if os.path.exists(fusion_gate_path):
                self.fusion_gate.load_state_dict(
                    torch.load(
                        fusion_gate_path,
                        weights_only=False,
                        map_location=torch.device(device),
                    )
                )
            else:
                print("No saved fusion_gate found, keeping initialized weights")

            # 兼容性加载瓶颈线性层参数（如果存在）
            dec_to_ctx_path = os.path.join(save_dir, "dec_to_ctx.pt")
            if os.path.exists(dec_to_ctx_path):
                sd = torch.load(dec_to_ctx_path, weights_only=False, map_location=torch.device(device))
                w = sd.get("weight", None)
                if w is None or w.shape == self.dec_to_ctx.weight.shape:
                    self.dec_to_ctx.load_state_dict(sd)
                else:
                    print(
                        f"Skip loading dec_to_ctx: weight shape {w.shape if w is not None else 'unknown'} vs {tuple(self.dec_to_ctx.weight.shape)}"
                    )
            enc_to_ctx_path = os.path.join(save_dir, "enc_to_ctx.pt")
            if os.path.exists(enc_to_ctx_path):
                sd = torch.load(enc_to_ctx_path, weights_only=False, map_location=torch.device(device))
                w = sd.get("weight", None)
                if w is None or w.shape == self.enc_to_ctx.weight.shape:
                    self.enc_to_ctx.load_state_dict(sd)
                else:
                    print(
                        f"Skip loading enc_to_ctx: weight shape {w.shape if w is not None else 'unknown'} vs {tuple(self.enc_to_ctx.weight.shape)}"
                    )
            ctx_to_model_path = os.path.join(save_dir, "ctx_to_model.pt")
            if os.path.exists(ctx_to_model_path):
                sd = torch.load(ctx_to_model_path, weights_only=False, map_location=torch.device(device))
                w = sd.get("weight", None)
                if w is None or w.shape == self.ctx_to_model.weight.shape:
                    self.ctx_to_model.load_state_dict(sd)
                else:
                    print(
                        f"Skip loading ctx_to_model: weight shape {w.shape if w is not None else 'unknown'} vs {tuple(self.ctx_to_model.weight.shape)}"
                    )
            # 读取head层号元数据
            with open(os.path.join(save_dir, "metadata.json"), "r") as f:
                metadata = json.load(f)
                self.layer_for_head = metadata["layer_for_head"]

    def train(self, mode: Optional[bool] = True):
        # 训练时只训练自定义head、附加解码器和音频特征提取模块，Whisper主干冻结
        self.whisper_model.eval()
        # 冻结Whisper参数
        for param in self.whisper_model.parameters():
            param.requires_grad = False
            
        for param in self.classifier.parameters():
            param.requires_grad = True
            
        for param in self.fusion_gate.parameters():
            param.requires_grad = True

        self.additional_decoder_block.train()
        self.stress_encoder.train()
        self.dec_to_ctx.train()
        self.enc_to_ctx.train()
        self.ctx_to_model.train()
        self.audio_feature_extractor.train()
        self.fusion_gate.train()
        self.classifier.train()

    def eval(self):
        self.whisper_model.eval()
        self.additional_decoder_block.eval()
        self.stress_encoder.eval()
        self.dec_to_ctx.eval()
        self.enc_to_ctx.eval()
        self.ctx_to_model.eval()
        self.audio_feature_extractor.eval()
        self.fusion_gate.eval()
        self.classifier.eval()

    def forward(
        self,
        input_features,
        attention_mask=None,
        decoder_input_ids=None,
        labels_head=None,
        whisper_labels=None,
    ):
        # 前向传播，返回自定义输出结构
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model.eval()

        # 1. 通过Whisper主干获取输出和隐藏状态
        backbone_outputs = self.whisper_model(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            labels=whisper_labels,
        )

        # 2. 获取decoder指定层隐藏状态；StressEncoder仅作为辅助分支
        decoder_last_layer_hidden_states = backbone_outputs.decoder_hidden_states[
            self.layer_for_head
        ].to(device)
        
        # 获取Whisper Encoder指定层的输出作为StressEncoder的输入
        # backbone_outputs.encoder_hidden_states 包含所有层的输出 (tuple)
        # 索引 0 是 embedding output, 1 是 layer 1 output, ..., 12 是 layer 12 output
        # 注意：transformers的output_hidden_states=True通常返回 (initial_embeddings, layer_1, ..., layer_N)
        # 所以第N层的输出索引通常是 N (如果从1开始数) 或者 N (如果包含embedding在0)
        # 实际上 encoder_hidden_states[i] 对应第 i 层 block 的输出 (i>=1)，encoder_hidden_states[0] 是 embedding
        # 我们使用 self.stress_encoder_input_layer 直接索引
        
        # 确保索引不越界
        layer_idx = self.stress_encoder_input_layer
        if layer_idx >= len(backbone_outputs.encoder_hidden_states):
            layer_idx = -1 # Use last layer
            
        stress_encoder_input = backbone_outputs.encoder_hidden_states[layer_idx].to(device)
        
        # StressEncoder 现在是一个简单的 TransformerEncoder，直接返回 tensor
        stress_encoder_hidden_states = self.stress_encoder(stress_encoder_input)
        
        # Apply dropout to StressEncoder output
        stress_encoder_hidden_states = self.stress_dropout(stress_encoder_hidden_states)
        
        # 获取用于Additional Decoder的Whisper Encoder输出
        enc_layer_idx = self.decoder_input_layer
        if enc_layer_idx >= len(backbone_outputs.encoder_hidden_states):
            enc_layer_idx = -1
        decoder_cross_attn_input = backbone_outputs.encoder_hidden_states[enc_layer_idx].to(device)

        # 3. 通过自定义解码器层
        additional_decoder_block_outputs = self.additional_decoder_block(
            hidden_states=decoder_last_layer_hidden_states,
            # 恢复使用 Whisper encoder 作为跨注意力的 K/V
            encoder_hidden_states=decoder_cross_attn_input,
        )
        # 3.1 使用可训练的跨注意力从encoder提取音频上下文特征（瓶颈映射到 d_ctx）
        dec_ctx = self.dec_to_ctx(decoder_last_layer_hidden_states)
        enc_ctx = self.enc_to_ctx(stress_encoder_hidden_states)
        audio_context, _ = self.audio_feature_extractor(
            query=dec_ctx,
            key=enc_ctx,
            value=enc_ctx,
        )
        # 3.2 将注意力输出回投影到 d_model 后与原路径拼接
        audio_context_model = self.ctx_to_model(audio_context)
        # audio_context_model = audio_context
        
        # 使用门控残差融合
        combined_features = self.fusion_gate(
            main=additional_decoder_block_outputs[0],
            aux=audio_context_model
        )
        
        head_logits = self.classifier(combined_features.to(device))

        # 4. softmax得到概率，argmax得到预测
        head_probs = F.softmax(head_logits, dim=-1)
        preds = head_probs.argmax(dim=-1).to(device)
        # mask掉无效标签
        if labels_head is not None:
            preds = torch.where(
                torch.isin(
                    labels_head, torch.tensor(list([-100])).to(device)  # 50257, 50362,
                ),
                torch.tensor(-100),
                preds,
            )
        # 5. 计算损失
        loss = None
        if labels_head is not None:
            # CrossEntropyLoss for the custom head
            loss = self.loss_fct(
                head_logits.reshape(-1, head_logits.size(-1)), labels_head.reshape(-1)
            )
            
            # [Constraint] Add L2 regularization to the StressEncoder branch output
            # This discourages the model from over-relying on the StressEncoder features
            # Coefficient is a hyperparameter to tune
            if self.stress_reg_coeff > 0:
                stress_reg = torch.mean(audio_context_model ** 2)
                loss += self.stress_reg_coeff * stress_reg
        return CustomModelOutput(
            logits=head_logits,
            labels_head=labels_head,
            whisper_logits=backbone_outputs.logits,
            loss=loss,
            preds=preds,
        )

    def generate(
        self,
        input_features,
        max_length=128,
        labels_head=None,
        whisper_labels=None,
        **generate_kwargs,
    ):
        """
        Generate both the Whisper output and custom head output sequences in alignment.
        """
        # 推理时同时生成Whisper输出和自定义head输出
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # 1. Whisper主干生成token序列
        whisper_outputs = self.whisper_model.generate(
            input_features=input_features,
            max_length=max_length,
            labels=whisper_labels,
            do_sample=False,
            **generate_kwargs,
        )

        # 2. 用生成的token序列做decoder输入，获取隐藏状态
        backbone_outputs = self.whisper_model(
            input_features=input_features,
            decoder_input_ids=whisper_outputs,
            output_hidden_states=True,
        )

        decoder_last_layer_hidden_states = backbone_outputs.decoder_hidden_states[
            self.layer_for_head
        ].to(device)
        
        # 获取Whisper Encoder指定层的输出作为StressEncoder的输入
        layer_idx = self.stress_encoder_input_layer
        if layer_idx >= len(backbone_outputs.encoder_hidden_states):
            layer_idx = -1
        stress_encoder_input = backbone_outputs.encoder_hidden_states[layer_idx].to(device)
        
        stress_encoder_hidden_states = self.stress_encoder(stress_encoder_input)
        # Apply dropout to StressEncoder output
        stress_encoder_hidden_states = self.stress_dropout(stress_encoder_hidden_states)

        # 获取用于Additional Decoder的Whisper Encoder输出
        enc_layer_idx = self.decoder_input_layer
        if enc_layer_idx >= len(backbone_outputs.encoder_hidden_states):
            enc_layer_idx = -1
        decoder_cross_attn_input = backbone_outputs.encoder_hidden_states[enc_layer_idx].to(device)

        # 3. 通过自定义解码器层（KV 使用 Whisper encoder）
        additional_decoder_block_outputs = self.additional_decoder_block(
            hidden_states=decoder_last_layer_hidden_states,
            encoder_hidden_states=decoder_cross_attn_input,
        )
        # 3.1 瓶颈映射到 d_ctx 后进行可训练跨注意力，再回投影到 d_model
        dec_ctx = self.dec_to_ctx(decoder_last_layer_hidden_states)
        enc_ctx = self.enc_to_ctx(stress_encoder_hidden_states)
        audio_context, _ = self.audio_feature_extractor(
            query=dec_ctx,
            key=enc_ctx,
            value=enc_ctx,
        )
        audio_context_model = self.ctx_to_model(audio_context)
        # 3.2 拼接两个路径的特征后送入分类头
        combined_features = self.fusion_gate(
            main=additional_decoder_block_outputs[0],
            aux=audio_context_model
        )
        head_logits = self.classifier(combined_features.to(device))
        # 4. 得到预测
        head_probs = F.softmax(head_logits, dim=-1)
        preds = head_probs.argmax(dim=-1).to(device)
        # mask掉无效token
        preds = torch.where(
            torch.isin(
                whisper_outputs.sequences, torch.tensor(list([50256])).to(device)  # 50257, 50362,
            ),
            torch.tensor(-100),
            preds,
        )
        return preds

    def generate_dual(
        self,
        input_features,
        attention_mask=None,
        max_length=200,
        labels_head=None,
        whisper_labels=None,
        **generate_kwargs,
    ):
        """
        Generate both the Whisper output and custom head output sequences in alignment.
        """
        # 推理时同时生成Whisper输出和自定义head输出，返回更丰富结构
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # 1. Whisper主干生成token序列
        whisper_outputs = self.whisper_model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            max_length=max_length,
            labels=whisper_labels,
            return_dict_in_generate=True,
            **generate_kwargs,
        )

        # 2. 用生成的token序列做decoder输入，获取隐藏状态
        backbone_outputs = self.whisper_model(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=whisper_outputs.sequences,
            output_hidden_states=True,
        )

        decoder_last_layer_hidden_states = backbone_outputs.decoder_hidden_states[
            self.layer_for_head
        ].to(device)
        
        # 获取Whisper Encoder指定层的输出作为StressEncoder的输入
        layer_idx = self.stress_encoder_input_layer
        if layer_idx >= len(backbone_outputs.encoder_hidden_states):
            layer_idx = -1
        stress_encoder_input = backbone_outputs.encoder_hidden_states[layer_idx].to(device)
        
        stress_encoder_hidden_states = self.stress_encoder(stress_encoder_input)
        # Apply dropout to StressEncoder output
        stress_encoder_hidden_states = self.stress_dropout(stress_encoder_hidden_states)

        # 获取用于Additional Decoder的Whisper Encoder输出
        enc_layer_idx = self.decoder_input_layer
        if enc_layer_idx >= len(backbone_outputs.encoder_hidden_states):
            enc_layer_idx = -1
        decoder_cross_attn_input = backbone_outputs.encoder_hidden_states[enc_layer_idx].to(device)

        # 3. 通过自定义解码器层
        additional_decoder_block_outputs = self.additional_decoder_block(
            hidden_states=decoder_last_layer_hidden_states,
            # KV 使用 Whisper encoder，StressEncoder 走辅助分支
            encoder_hidden_states=decoder_cross_attn_input,
        )
        # 3.1 使用可训练的跨注意力从encoder提取音频上下文特征（瓶颈映射到 d_ctx）
        dec_ctx = self.dec_to_ctx(decoder_last_layer_hidden_states)
        enc_ctx = self.enc_to_ctx(stress_encoder_hidden_states)
        audio_context, _ = self.audio_feature_extractor(
            query=dec_ctx,
            key=enc_ctx,
            value=enc_ctx,
        )
        # 3.2 将注意力输出回投影到 d_model 后与原路径拼接
        audio_context_model = self.ctx_to_model(audio_context)
        combined_features = self.fusion_gate(
            main=additional_decoder_block_outputs[0],
            aux=audio_context_model
        )
        head_logits = self.classifier(combined_features.to(device))
        head_probs = F.softmax(head_logits, dim=-1)
        preds = head_probs.argmax(dim=-1).to(device)
        # mask掉无效token
        preds = torch.where(
            torch.isin(
                whisper_outputs.sequences, torch.tensor(list([50256])).to(device)  # 50257, 50362,
            ),
            torch.tensor(-100),
            preds,
        )
        return CustomModelOutput(
            logits=head_logits,
            head_preds=preds,
            whisper_logits=whisper_outputs.logits,
            preds=whisper_outputs.sequences
        )

    def __str__(self):
        return "WhiStress"
