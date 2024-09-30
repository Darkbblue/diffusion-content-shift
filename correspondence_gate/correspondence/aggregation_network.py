import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class AggregationNetwork(nn.Module):
    def __init__(
            self,
            dim,
            attn_dim,
            weight_num,
        ):
        super().__init__()
        self.weight_assigner = []  # bucket x 1
        for _ in range(weight_num):
            self.weight_assigner.append(nn.Sequential(
                # nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
                # nn.ReLU(),
                # nn.BatchNorm2d(dim // 2),
                # nn.Dropout(),

                # nn.Conv2d(dim // 2, 128, kernel_size=3, stride=2, padding=1, bias=False),
                # nn.ReLU(),
                # nn.BatchNorm2d(128),
                # nn.Dropout(),

                nn.Conv2d(dim, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.Dropout(),

                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten(),  # just for reshape
                nn.Linear(128, dim),
            ))
        self.weight_assigner = nn.ModuleList(self.weight_assigner)

        self.out = [nn.Conv2d(dim+attn_dim, dim+attn_dim, kernel_size=3, stride=1, padding=1, bias=False)] * weight_num
        self.out = nn.ModuleList(self.out)

        self.similarity_penalty_base = weight_num * (weight_num) * 0.5
        self.sparsity_penalty_base = weight_num

        # For CLIP symmetric cross entropy loss during training
        self.logit_scale = torch.ones([]) * np.log(1 / 0.07)

        # record data for weight distribution analysis
        # self.to_store = []

    def forward(self, x, attns, do_conv):
        # x: batch x bucket x dim x w x h
        # assign weights
        linear_combination = []
        weight_save = []
        if not do_conv:
            for m in self.weight_assigner:
                weights = torch.stack([m(x[:,b,:]) for b in range(x.shape[1])])  # bucket x batch x 1
                weights = F.softmax(weights, dim=0).permute(1, 0, 2)  # batch x bucket x 1
                weight_save.append(weights)
                weights = weights.unsqueeze(-1).unsqueeze(-1)
                feat = torch.cat([torch.sum(weights * x, dim=1), attns], dim=1)
                linear_combination.append(feat)
        else:
            for m, o in zip(self.weight_assigner, self.out):
                weights = torch.stack([m(x[:,b,:]) for b in range(x.shape[1])])  # bucket x batch x 1
                weights = F.softmax(weights, dim=0).permute(1, 0, 2)  # batch x bucket x 1
                weight_save.append(weights)
                weights = weights.unsqueeze(-1).unsqueeze(-1)
                feat = torch.cat([torch.sum(weights * x, dim=1), attns], dim=1)
                feat = o(feat)
                linear_combination.append(feat)
        x = torch.concatenate(linear_combination, dim=1)  # batch x (dim * weight_num) x w x h

        # record data for weight distribution analysis
        # for b in range(2):
        #     store_entry = []
        #     for w in weight_save:
        #         store_entry.append([w[b,i].detach().cpu().item() for i in range(6)])
        #     self.to_store.append(store_entry)

        # extra restriction
        similarity_penalty = 0.
        sparsity_penalty = 0.
        for i in range(len(weight_save)):
            sparsity_penalty += torch.linalg.norm(weight_save[i], dim=1).mean()
        similarity_penalty /= self.similarity_penalty_base
        sparsity_penalty /= self.sparsity_penalty_base
        return x, -sparsity_penalty, -similarity_penalty
