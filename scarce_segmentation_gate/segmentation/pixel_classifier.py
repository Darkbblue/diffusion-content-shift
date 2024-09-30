import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter

from torch.distributions import Categorical
from .utils import colorize_mask, oht_to_scalar
from .data_util import get_palette, get_class_names
from PIL import Image


# Adopted from https://github.com/nv-tlabs/datasetGAN_release/blob/d9564d4d2f338eaad78132192b865b6cc1e26cac/datasetGAN/train_interpreter.py#L68
class pixel_classifier(nn.Module):
    def __init__(self, numpy_class, dim, dim_ddpm, weight_num):
        super(pixel_classifier, self).__init__()
        # bucket x dim
        self.weight_assigner = []  # bucket x 1
        for _ in range(weight_num):
            self.weight_assigner.append(nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Dropout(),
                # nn.Linear(128, 1),
                nn.Linear(128, dim),
                # nn.Dropout(),
            ))
        self.weight_assigner = nn.ModuleList(self.weight_assigner)

        self.weight_assigner_ddpm = []  # bucket x 1
        for _ in range(weight_num):
            self.weight_assigner_ddpm.append(nn.Sequential(
                nn.Linear(dim_ddpm, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Dropout(),
                # nn.Linear(128, 1),
                nn.Linear(128, dim_ddpm),
                # nn.Dropout(),
            ))
        self.weight_assigner_ddpm = nn.ModuleList(self.weight_assigner_ddpm)

        dim = dim * weight_num
        if numpy_class < 30:
            self.layers = nn.Sequential(
                nn.Linear(dim+dim_ddpm, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=32),
                nn.Linear(32, numpy_class)
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(dim+dim_ddpm, 256),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=256),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, numpy_class)
            )

        self.similarity_penalty_base = weight_num * (weight_num) * 0.5
        self.sparsity_penalty_base = weight_num

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, x, ddpm):
        # x: batch x bucket x dim
        # assign weights
        linear_combination = []
        weight_save = []
        for i, m in enumerate(self.weight_assigner):
            weights = torch.stack([m(x[:,b,:]) for b in range(x.shape[1])])  # bucket x batch x 1
            weights = F.softmax(weights, dim=0).permute(1, 0, 2)  # batch x bucket x 1
            # weight_save.append(weights.squeeze(-1))
            weight_save.append(weights)
            linear_combination.append(torch.sum(weights * x, dim=1))  # batch x dim
        x = torch.concatenate(linear_combination, dim=1)  # batch x (dim * weight_num)

        # ddpm: batch x dim_ddpm
        linear_combination = []
        weight_save_ddpm = []
        for i, m in enumerate(self.weight_assigner_ddpm):
            weights = torch.stack([m(ddpm[:,b,:]) for b in range(ddpm.shape[1])])  # bucket x batch x 1
            weights = F.softmax(weights, dim=0).permute(1, 0, 2)  # batch x bucket x 1
            weight_save_ddpm.append(weights)
            linear_combination.append(torch.sum(weights * ddpm, dim=1))  # batch x dim
        ddpm = torch.concatenate(linear_combination, dim=1)  # batch x (dim * weight_num)
        x = torch.concatenate([x, ddpm], dim=1)

        # classification
        x = self.layers(x)

        # extra restriction
        similarity_penalty = torch.zeros(1, device=x.device)#0.
        sparsity_penalty = 0.
        for i in range(len(weight_save)):
            sparsity_penalty += torch.linalg.norm(weight_save[i], dim=1).mean()
        for i in range(len(weight_save_ddpm)):
            sparsity_penalty += torch.linalg.norm(weight_save_ddpm[i], dim=1).mean()
        similarity_penalty /= self.similarity_penalty_base
        sparsity_penalty /= self.sparsity_penalty_base
        return x, - sparsity_penalty, - similarity_penalty


def predict_labels(models, features, ddpm, size):
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    
    mean_seg = None
    all_seg = []
    all_entropy = []
    seg_mode_ensemble = []

    softmax_f = nn.Softmax(dim=1)
    with torch.no_grad():
        for MODEL_NUMBER in range(len(models)):
            bc = 256 * 8
            fi = 0
            pp = []
            while fi < (features.shape[0]):
                ffe = features[fi:fi+bc,:,:]
                dd = ddpm[fi:fi+bc,:]
                preds, _, _ = models[MODEL_NUMBER](ffe.cuda(), dd.cuda())
                pp.append(preds)
                fi += bc
            preds = torch.concatenate(pp, dim=0)
            # preds, _, _ = models[MODEL_NUMBER](features.cuda(), ddpm.cuda())
            entropy = Categorical(logits=preds).entropy()
            all_entropy.append(entropy)
            all_seg.append(preds)

            if mean_seg is None:
                mean_seg = softmax_f(preds)
            else:
                mean_seg += softmax_f(preds)

            img_seg = oht_to_scalar(preds)
            img_seg = img_seg.reshape(*size)
            img_seg = img_seg.cpu().detach()

            seg_mode_ensemble.append(img_seg)

        mean_seg = mean_seg / len(all_seg)

        full_entropy = Categorical(mean_seg).entropy()

        js = full_entropy - torch.mean(torch.stack(all_entropy), 0)
        top_k = js.sort()[0][- int(js.shape[0] / 10):].mean()

        img_seg_final = torch.stack(seg_mode_ensemble, dim=-1)
        img_seg_final = torch.mode(img_seg_final, 2)[0]
    return img_seg_final, top_k


def save_predictions(args, image_paths, preds):
    palette = get_palette(args['category'])
    os.makedirs(os.path.join(args['exp_dir'], 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(args['exp_dir'], 'visualizations'), exist_ok=True)

    for i, pred in enumerate(preds):
        filename = image_paths[i].split('/')[-1].split('.')[0]
        pred = np.squeeze(pred)
        np.save(os.path.join(args['exp_dir'], 'predictions', filename + '.npy'), pred)

        mask = colorize_mask(pred, palette)
        Image.fromarray(mask).save(
            os.path.join(args['exp_dir'], 'visualizations', filename + '.jpg')
        )


def compute_iou(args, preds, gts, print_per_class_ious=True):
    class_names = get_class_names(args['category'])

    ids = range(args['number_class'])

    unions = Counter()
    intersections = Counter()

    for pred, gt in zip(preds, gts):
        for target_num in ids:
            if target_num == args['ignore_label']: 
                continue
            preds_tmp = (pred == target_num).astype(int)
            gts_tmp = (gt == target_num).astype(int)
            unions[target_num] += (preds_tmp | gts_tmp).sum()
            intersections[target_num] += (preds_tmp & gts_tmp).sum()
    
    ious = []
    for target_num in ids:
        if target_num == args['ignore_label']: 
            continue
        iou = intersections[target_num] / (1e-8 + unions[target_num])
        ious.append(iou)
        if print_per_class_ious:
            print(f"IOU for {class_names[target_num]} {iou:.4}")
    return np.array(ious).mean()


def load_ensemble(args, device='cpu'):
    models = []
    for i in range(args['model_num']):
        model_path = os.path.join(args['exp_dir'], f'model_{i}.pth')
        state_dict = torch.load(model_path)['model_state_dict']
        model = nn.DataParallel(pixel_classifier(args["number_class"], args['dim'][-1], dim_ddpm=args['ddpm_feature_len'], weight_num=args['weight_num']))
        model.load_state_dict(state_dict)
        model = model.module.to(device)
        models.append(model.eval())
    return models
