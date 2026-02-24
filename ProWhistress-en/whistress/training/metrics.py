# metrics.py 该文件实现了WhiStress模型的评估指标计算，支持准确率、精确率、召回率和F1
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 评估指标类，封装了多种常用指标
class WhiStressMetrics:
    def __init__(self):
        # 使用本地sklearn指标，不依赖在线evaluate库
        pass

    def compute_metrics(self, pred):
        # 忽略被mask掉的标签（-100），只对有效部分计算指标
        def ignore_masked_predictions(pred_ids, label_ids, pad_token_id):
            # Create a mask where label_ids is not equal to pad_token_id
            mask_label_ids = label_ids != pad_token_id

            # Flatten the tensors to process them as one-dimensional arrays
            pred_ids_flat = pred_ids[mask_label_ids].flatten()
            label_ids_flat = label_ids[mask_label_ids].flatten()

            return pred_ids_flat, label_ids_flat

        pred_ids = pred["predictions"]
        label_ids = pred["label_ids"]
        preds, labels = ignore_masked_predictions(pred_ids, label_ids, -100)

        metrics = {}
        # 计算准确率
        metrics["accuracy"] = accuracy_score(labels, preds)
        
        # 计算精确率
        metrics["precision"] = precision_score(labels, preds, pos_label=1, zero_division=0)
        
        # 计算召回率
        metrics["recall"] = recall_score(labels, preds, pos_label=1, zero_division=0)
        
        # 计算F1分数
        metrics["f1"] = f1_score(labels, preds, pos_label=1, zero_division=0)

        return metrics
