import torch
import torch.nn as nn


class Ens(nn.Module):
    def __init__(self, models):
        super(Ens, self).__init__()
        self.num_models = len(models)
        self.model_1 = models[0]
        self.model_2 = models[1]
        self.model_3 = models[2]
        self.model_4 = models[3]
        self.models = [self.model_1, self.model_2, self.model_3, self.model_4]

    def forward(self, x):
        logits = []
        for i in range(self.num_models):
            logits.append(self.models[i](x))
        logit = torch.stack(logits)
        return logit.mean(dim=0)



