import torch
import torch.nn.functional as F
import torch.nn as nn
from .ditill_utils import *
from copy import deepcopy

from .losses import mape_loss, mase_loss, smape_loss

class CosineSimilarityLoss(nn.Module):
    """
    Cosine Distance Loss for semantic representation alignment.
    Loss = 1 - cosine_similarity(x1, x2). Ranges from 0 (identical) to 2 (opposite).
    Ideal for Vision Transformer (DINO-V3) intermediate feature distillation.
    """
    def __init__(self, dim=-1):
        super(CosineSimilarityLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=dim)

    def forward(self, x1, x2):
        if x1.shape != x2.shape:
            # Flatten non-batch dimensions if shape mismatch
            x1 = x1.reshape(x1.size(0), -1)
            x2 = x2.reshape(x2.size(0), -1)
        return torch.mean(1.0 - self.cos(x1, x2))


class KLDivergenceLoss(nn.Module):
    """
    Kullback-Leibler Divergence Loss for soft probability logits distillation.
    """
    def __init__(self, temperature=2.0):
        super(KLDivergenceLoss, self).__init__()
        self.temp = temperature
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, *args, **kwargs):
        p = F.log_softmax(student_logits / self.temp, dim=-1)
        q = F.softmax(teacher_logits / self.temp, dim=-1)
        return self.kl(p, q) * (self.temp ** 2)


class ContrastiveFeatureLoss(nn.Module):
    """
    InfoNCE Contrastive Loss for cross-modal alignment between time series
    and teacher (visual ViT / text LLM) representations.
    Attracts matched time-series/teacher pairs while repelling unmatched samples.
    """
    def __init__(self, temperature=0.07):
        super(ContrastiveFeatureLoss, self).__init__()
        self.temp = temperature

    def forward(self, feat_time, feat_teacher):
        z_time = F.normalize(feat_time.reshape(feat_time.size(0), -1), dim=-1)
        z_teacher = F.normalize(feat_teacher.reshape(feat_teacher.size(0), -1), dim=-1)
        sim_matrix = torch.matmul(z_time, z_teacher.T) / self.temp
        labels = torch.arange(feat_time.size(0), device=feat_time.device)
        return F.cross_entropy(sim_matrix, labels)


loss_dict = {
    "l1": nn.L1Loss(),
    "smooth_l1": nn.SmoothL1Loss(),
    "huber": nn.HuberLoss(),
    "ce": nn.CrossEntropyLoss(),
    "mse": nn.MSELoss(),
    "cosine": CosineSimilarityLoss(dim=-1),
    "kl_div": KLDivergenceLoss(temperature=2.0),
    "contrastive": ContrastiveFeatureLoss(temperature=0.07),
    "smape": smape_loss(),
    "mape": mape_loss(),
    "mase": mase_loss(),
}

class DistillationLoss(nn.Module):
    """
    Tripartite Cross-Modal Distillation Loss for CALF (supporting GPT-2 & DINO-V3):
      L_total = w_task * L_task + w_logits * L_logits + w_feat * L_feat
      
      1. Task Loss (L_task): Supervised target prediction loss (L1, MSE, Smooth-L1, SMAPE, MASE).
      2. Logits Loss (L_logits): Teacher-student output alignment (L1, MSE, KL-Div).
      3. Feature Loss (L_feat): Multi-layer intermediate representation alignment across
         time series student layers and visual/text teacher layers with exponential depth weighting.
    """
    def __init__(self, distill_loss, logits_loss, task_loss, task_name, feature_w=0.01, logits_w=1.0, task_w=1.0, decay_rate=0.8):
        super(DistillationLoss, self).__init__()
        
        self.task_w = task_w
        self.logits_w = logits_w
        self.feature_w = feature_w
        self.decay_rate = decay_rate
        
        print(f"DistillationLoss Config: task_w={task_w}, logits_w={logits_w}, feature_w={feature_w}, decay={decay_rate}")
        print(f"  - Distill Loss (Feature): {distill_loss}")
        print(f"  - Logits Loss (Output) : {logits_loss}")
        print(f"  - Task Loss (Target)   : {task_loss}")
        
        self.distill_loss_name = distill_loss
        self.logits_loss_name = logits_loss
        self.task_loss_name = task_loss
        
        self.feature_loss = loss_dict.get(distill_loss, nn.L1Loss())
        self.logits_loss = loss_dict.get(logits_loss, nn.L1Loss())
        self.task_loss = loss_dict.get(task_loss, nn.L1Loss())
        
        self.task_name = task_name

    def forward(self, outputs, batch_y, in_sample=None, freq_map=None, batch_y_mark=None):
        """
        outputs_time: Predictions from student time-series backbone
        outputs_text: Predictions from teacher model (LLM / DINO-V3 ViT)
        intermidiate_feat_time: Intermediate hidden states [num_layers + 1] from student
        intermidiate_feat_text: Intermediate hidden states [num_layers + 1] from teacher
        """
        outputs_text = outputs.get("outputs_text")
        outputs_time = outputs.get("outputs_time")
        intermidiate_feat_time = outputs.get("intermidiate_time")
        intermidiate_feat_text = outputs.get("intermidiate_text")
        
        # 1. Multi-layer Intermediate Feature Distillation Loss (Exponential Depth Decay)
        if intermidiate_feat_time is not None and intermidiate_feat_text is not None and len(intermidiate_feat_time) > 0:
            feature_loss = sum(
                [
                    (self.decay_rate ** idx) * self.feature_loss(feat_time, feat_text)
                    for idx, (feat_time, feat_text) in enumerate(
                        zip(intermidiate_feat_time[::-1], intermidiate_feat_text[::-1])
                    )
                ]
            )
        else:
            feature_loss = 0.0

        # 2. Output Layer Teacher-Student Logits Distillation Loss
        if outputs_text is not None: 
            if self.task_name in ["long_term_forecast", "classification", "imputation", "anomaly_detection"]:
                logits_loss = self.logits_loss(outputs_time, outputs_text)
            elif self.task_name == "short_term_forecast":
                logits_loss = self.logits_loss(in_sample, freq_map, outputs_time, outputs_text, batch_y_mark)
            else:
                logits_loss = self.logits_loss(outputs_time, outputs_text)
        else:
            logits_loss = 0.0
            
        # 3. Supervised Task Target Loss
        batch_y = batch_y.to(outputs_time.device)
        
        if self.task_name == "long_term_forecast":
            task_loss = self.task_loss(outputs_time, batch_y)
        elif self.task_name == "short_term_forecast":
            task_loss = self.task_loss(in_sample, freq_map, outputs_time, batch_y, batch_y_mark)
        elif self.task_name in ["classification", "imputation", "anomaly_detection"]:
            task_loss = self.task_loss(outputs_time, batch_y)
        else:
            task_loss = self.task_loss(outputs_time, batch_y)

        # 4. Total Combined Objective Loss
        total_loss = self.task_w * task_loss + self.logits_w * logits_loss + self.feature_w * feature_loss
        return total_loss
