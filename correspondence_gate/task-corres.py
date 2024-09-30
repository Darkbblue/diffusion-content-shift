import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import random
from tqdm import tqdm

from correspondence.correspondence_utils import (
    load_image_pair,
    batch_cosine_sim,
    points_to_idxs,
    find_nn_source_correspondences,
    draw_correspondences,
    compute_pck,
    rescale_points
)
from correspondence.stable_diffusion.resnet import collect_dims
from correspondence.aggregation_network import AggregationNetwork

device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")

logging = None
def log(s):
    global logging
    print(s)
    logging.write(s+'\n')
    logging.flush()


def get_rescale_size(config):
    return (64, 64), (512, 512)


def log_aggregation_network(aggregation_network, config):
    mixing_weights = torch.nn.functional.softmax(aggregation_network.mixing_weights)
    num_layers = len(aggregation_network.feature_dims)
    num_timesteps = len(aggregation_network.save_timestep)
    save_timestep = aggregation_network.save_timestep
    if config["diffusion_mode"] == "inversion":
        save_timestep = save_timestep[::-1]
    fig, ax = plt.subplots()
    ax.imshow(mixing_weights.view((num_timesteps, num_layers)).T.detach().cpu().numpy())
    ax.set_ylabel("Layer")
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels(range(1, num_layers+1))
    ax.set_xlabel("Timestep")
    ax.set_xticklabels(save_timestep)
    ax.set_xticks(range(num_timesteps))
    # wandb.log({f"mixing_weights": plt})


def get_hyperfeats(args, aggregation_network, src_path, tgt_path, category):
    f_src = []
    f_tgt = []
    for sub_feature in args.feature_id[:-1]:
        src = os.path.join(args.feature_path, sub_feature, src_path.replace('jpg', 'npy'))
        tgt = os.path.join(args.feature_path, sub_feature, tgt_path.replace('jpg', 'npy'))
        src, tgt = torch.from_numpy(np.load(src)).to(device), torch.from_numpy(np.load(tgt)).to(device)
        src, tgt = src.float(), tgt.float()
        f_src.append(src)
        f_tgt.append(tgt)
    f_src = torch.stack(f_src)  # bucket x c x w x h
    f_tgt = torch.stack(f_tgt)
    feats = torch.stack([f_src, f_tgt])  # 2 x bucket x c x w x h

    # attention
    src = os.path.join(args.feature_path, args.feature_id[-1], src_path.replace('jpg', 'npy'))
    tgt = os.path.join(args.feature_path, args.feature_id[-1], tgt_path.replace('jpg', 'npy'))
    src, tgt = torch.from_numpy(np.load(src)).to(device), torch.from_numpy(np.load(tgt)).to(device)
    src, tgt = src.float(), tgt.float()  # c x w x h
    attns = torch.stack([src, tgt])  # 2 x c x w x h

    diffusion_hyperfeats, sparsity, diversity = aggregation_network(feats, attns, do_conv=args.algorithm=='conv')
    img1_hyperfeats = diffusion_hyperfeats[0][None, ...]
    img2_hyperfeats = diffusion_hyperfeats[1][None, ...]
    return img1_hyperfeats, img2_hyperfeats, sparsity, diversity


def compute_clip_loss(aggregation_network, img1_hyperfeats, img2_hyperfeats, source_points, target_points, output_size):
    # Assumes hyperfeats are batch_size=1 to avoid complex indexing
    # Compute in both directions for cycle consistency
    source_logits = aggregation_network.logit_scale.exp() * batch_cosine_sim(img1_hyperfeats, img2_hyperfeats)
    target_logits = aggregation_network.logit_scale.exp() * batch_cosine_sim(img2_hyperfeats, img1_hyperfeats)
    source_idx = torch.from_numpy(points_to_idxs(source_points, output_size)).long().to(source_logits.device)
    target_idx = torch.from_numpy(points_to_idxs(target_points, output_size)).long().to(target_logits.device)
    loss_source = torch.nn.functional.cross_entropy(source_logits[0, source_idx], target_idx)
    loss_target = torch.nn.functional.cross_entropy(target_logits[0, target_idx], source_idx)
    loss = (loss_source + loss_target) / 2
    return loss


def save_model(config, aggregation_network, optimizer, step):
    dict_to_save = {
        "step": step,
        "config": config,
        "aggregation_network": aggregation_network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    results_folder = args.task_path
    torch.save(dict_to_save, f"{results_folder}/checkpoint_step_{step}.pt")


def validate(config, aggregation_network, val_anns):
    output_size, load_size = get_rescale_size(config)
    plot_every_n_steps = -1
    pck_threshold = 0.1
    ids, val_dist, val_pck_img, val_pck_bbox = [], [], [], []
    for j, ann in enumerate(tqdm(val_anns)):
        with torch.no_grad():
            source_points, target_points, src_path, tgt_path, category = load_image_pair(ann, load_size, device, image_path=config.image_path)
            img1_hyperfeats, img2_hyperfeats, _, _ = get_hyperfeats(config, aggregation_network, src_path, tgt_path, category)
            loss = compute_clip_loss(aggregation_network, img1_hyperfeats, img2_hyperfeats, source_points, target_points, output_size)
            # log(f'val/loss: {loss.item()}')
            # Log NN correspondences
            _, predicted_points = find_nn_source_correspondences(img1_hyperfeats, img2_hyperfeats, source_points, output_size, load_size)
            predicted_points = predicted_points.detach().cpu().numpy()
            # Rescale to the original image dimensions
            target_size = ann["target_size"]
            predicted_points = rescale_points(predicted_points, load_size, target_size)
            target_points = rescale_points(target_points, load_size, target_size)
            dist, pck_img, sample_pck_img = compute_pck(predicted_points, target_points, target_size, pck_threshold=pck_threshold)
            _, pck_bbox, sample_pck_bbox = compute_pck(predicted_points, target_points, target_size, pck_threshold=pck_threshold, target_bounding_box=ann["target_bounding_box"])
            val_dist.append(dist)
            val_pck_img.append(pck_img)
            val_pck_bbox.append(pck_bbox)
            ids.append([j] * len(dist))
            # if plot_every_n_steps > 0 and j % plot_every_n_steps == 0:
            #     title = f"pck@{pck_threshold}_img: {sample_pck_img.round(decimals=2)}"
            #     title += f"\npck@{pck_threshold}_bbox: {sample_pck_bbox.round(decimals=2)}"
            #     draw_correspondences(source_points, predicted_points, img1_pil, img2_pil, title=title, radius1=1)
    ids = np.concatenate(ids)
    val_dist = np.concatenate(val_dist)
    val_pck_img = np.concatenate(val_pck_img)
    val_pck_bbox = np.concatenate(val_pck_bbox)
    # df = pd.DataFrame({
    #     "id": ids,
    #     "distances": val_dist,
    #     "pck_img": val_pck_img,
    #     "pck_bbox": val_pck_bbox,
    # })
    log(f"val/pck_img: {val_pck_img.sum() / len(val_pck_img)}")
    log(f"val/pck_bbox: {val_pck_bbox.sum() / len(val_pck_bbox)}")
    # wandb.log({f"val/distances_csv": wandb.Table(dataframe=df)})
    return val_pck_img.sum() / len(val_pck_img), val_pck_bbox.sum() / len(val_pck_bbox)


def train(config, aggregation_network, optimizer, train_anns, val_anns):
    output_size, load_size = get_rescale_size(config)
    max1, max2 = 0, 0
    # heuristic optimization for sparsity
    last_sparsity = 0
    last_loss_delta = 0
    loss_log = None
    sparsity = 0
    for epoch in range(config.max_epochs):
        epoch_train_anns = np.random.permutation(train_anns)[:config.max_steps_per_epoch]
        for i, ann in enumerate(tqdm(epoch_train_anns)):
            step = epoch * len(epoch_train_anns) + i
            optimizer.zero_grad()
            source_points, target_points, src_path, tgt_path, category = load_image_pair(ann, load_size, device, image_path=config.image_path, output_size=output_size)
            img1_hyperfeats, img2_hyperfeats, sparsity_loss, diversity_loss = get_hyperfeats(config, aggregation_network, src_path, tgt_path, category)
            loss = compute_clip_loss(aggregation_network, img1_hyperfeats, img2_hyperfeats, source_points, target_points, output_size)
            # (loss + sparsity * config.sparsity + diversity * config.diversity).backward()
            (loss + sparsity * sparsity_loss * loss.detach() / (-sparsity_loss.detach())).backward()
            optimizer.step()
            # heuristic optimization for sparsity
            if (i+1) % 5 == 0:
                loss_delta = 0 if loss_log is None else loss_log - loss.item()
                loss_log = loss.item()
                ignore_comparision = random.choice([True] + [False] * 19)
                if ignore_comparision or loss_delta > last_loss_delta:
                # if loss_delta > last_loss_delta:
                    last_sparsity = sparsity
                    last_loss_delta = loss_delta
                last_loss_delta *= 0.9
                sparsity = max(0, last_sparsity + random.choice([-0.00001, 0, 0.00001]))

            if i % 100 == 0:
                log(f'epoch {epoch} step {step} train/loss: {loss.item()}')
            if step > 0 and step % 500 == 0:
                del img1_hyperfeats, img2_hyperfeats, sparsity_loss, diversity_loss, loss
                torch.cuda.empty_cache()
                with torch.no_grad():
                    # log_aggregation_network(aggregation_network, config)
                    save_model(config, aggregation_network, optimizer, step)
                    log('####################')
                    log(f'epoch {epoch} step {step}')
                    r1, r2 = validate(config, aggregation_network, val_anns)
                    max1 = max(max1, r1)
                    max2 = max(max2, r2)
    log(f'{max1}, {max2}')


def main(args):
    # set random seed
    if args.seed:
        print('Seed: ', args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # initialize logging
    global logging
    os.makedirs(args.task_path, exist_ok=True)
    logging = open(os.path.join(args.task_path, 'log.txt'), 'w')

    # load model
    aggregation_network = AggregationNetwork(args.feature_len, args.attn_len, weight_num=1).to(device)

    parameter_groups = [
        {"params": aggregation_network.parameters(), "lr": args.lr}
    ]
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    val_anns = json.load(open(args.val_path))
    if args.load_weight:
        aggregation_network.load_state_dict(torch.load(args.load_weight, map_location="cpu")["aggregation_network"])
        validate(args, aggregation_network, val_anns)
    else:
        # "The loss computation compute_clip_loss assumes batch_size=2."
        train_anns = json.load(open(args.train_path))
        train(args, aggregation_network, optimizer, train_anns, val_anns)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument("--config_path", type=str, help="Path to yaml config file", default="configs/train.yaml")
    # model param
    parser.add_argument('--seed', type=int,  default=None)
    # parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_epochs', type=int, default=2)
    parser.add_argument('--max_steps_per_epoch', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    # io
    parser.add_argument('--log_path', type=str, help='where to store logs and checkpoints')
    parser.add_argument('--dataset_path', type=str, default='datasets/SPair-71k/JPEGImages')
    parser.add_argument('--feature_path', type=str, help='where are diffusion features stored')
    parser.add_argument('--feature_id', type=str, nargs='+', help='features to be put in the bucket')
    # task info
    parser.add_argument('--task_name', type=str, help='logs and checkpoints will be saved at a corresponding folder')
    parser.add_argument('--feature_len', type=int)
    parser.add_argument('--attn_len', type=int)
    parser.add_argument('--algorithm', type=str, choices=('nn', 'conv'), default='conv')
    parser.add_argument('--load_weight', type=str, default=None)

    args = parser.parse_args()
    args.task_path = os.path.join(args.log_path, args.task_name)
    args.image_path = args.dataset_path
    args.train_path = os.path.join(args.dataset_path, 'spair_71k_train.json')
    args.val_path = os.path.join(args.dataset_path, 'spair_71k_val-360.json')
    args.test_path = os.path.join(args.dataset_path, 'spair_71k_test-6.json')
    main(args)
