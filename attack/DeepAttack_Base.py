"""
Base implement of DeepAttack
"""
import torch
import torch.nn as nn
from attack.Attack_Base import Attack_Base


class DeepAttack_Base(Attack_Base):
    def __init__(self, model, eps, alpha, iters, beta=1.0, decay=0.0, max_value=1., min_value=0, feat_extractor=None):
        super().__init__(model=model, eps=eps, max_value=max_value, min_value=min_value)
        assert feat_extractor is not None
        self.iters = iters
        self.alpha = alpha
        self.decay = decay
        self.beta = beta
        self.feat_extractor = feat_extractor

    def feat_loss(self, feats_clean, feats_adv, loss_func):
        loss = 0
        for feat_clean, feat_adv in zip(feats_clean, feats_adv):
            loss += loss_func(feat_clean, feat_adv)
        return loss

    def total_loss(self, data, feat_clean, label):
        # loss func
        ce_loss_func = nn.CrossEntropyLoss()
        mse_loss_func = nn.MSELoss()

        # output
        output_logits = self.model(data)
        output_feats = self.feat_extractor(data)

        # loss
        loss_ce = ce_loss_func(output_logits, label)
        loss_feat = self.feat_loss(output_feats, feat_clean, mse_loss_func)
        loss = loss_ce + self.beta * loss_feat

        return loss
