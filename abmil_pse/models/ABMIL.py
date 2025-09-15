import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionTanhSigmoidGating(nn.Module):
    def __init__(self, D=64, L=64, dropout=0.25):
        super(AttentionTanhSigmoidGating, self).__init__()
        self.tanhV = nn.Sequential(nn.Linear(D, L), nn.Tanh(), nn.Dropout(dropout))
        self.sigmU = nn.Sequential(nn.Linear(D, L), nn.Sigmoid(), nn.Dropout(dropout))
        self.w = nn.Linear(L, 1)

    def forward(self, H, return_raw_attention=False):
        A_raw = self.w(self.tanhV(H) * self.sigmU(H))
        A_norm = F.softmax(A_raw, dim=0)
        assert abs(A_norm.sum() - 1) < 1e-3
        if return_raw_attention:
            return A_norm, A_raw
        return A_norm


class ABMIL(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=64, dropout=0.25, n_classes=2):
        super().__init__()
        self.inst_level_fc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.global_attn = AttentionTanhSigmoidGating(L=hidden_dim, D=hidden_dim)
        self.bag_level_classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, X, return_raw_attention=False):
        H_inst = self.inst_level_fc(X)
        if return_raw_attention:
            A_norm, A_raw = self.global_attn(H_inst, return_raw_attention=True)
        else:
            A_norm = self.global_attn(H_inst)
        z = torch.sum(A_norm * H_inst, dim=0)
        logits = self.bag_level_classifier(z).unsqueeze(dim=0)
        if return_raw_attention:
            return logits, A_raw
        return logits, A_norm