"""
* Based on https://github.com/JosephKJ/OWOD
* Apache License
* Copyright (c) 2021, K J Joseph
* Copyright (c) SafeDNN group 2023
"""

import os
import numpy as np
import argparse
import mmcv
import tqdm
from mmcv import Config
from reliability.Fitters import Fit_Weibull_3P
import torch
from torch.distributions.weibull import Weibull
from torch.distributions.transforms import AffineTransform
from torch.distributions.transformed_distribution import TransformedDistribution


def create_distribution(scale, shape, shift):
    wd = Weibull(scale=scale, concentration=shape)
    transforms = AffineTransform(loc=shift, scale=1.)
    weibull = TransformedDistribution(wd, transforms)
    return weibull


def compute_prob(x, distribution):
    eps_radius = 0.5
    num_eval_points = 100
    x = torch.tensor(x)
    prob = torch.zeros_like(x)
    for i in range(prob.size().numel()):
        start_x = x[i] - eps_radius
        end_x = x[i] + eps_radius
        step = (end_x - start_x) / num_eval_points
        dx = torch.linspace(start_x, end_x, num_eval_points)
        try:
            pdf = distribution.log_prob(dx).exp()
            prob[i] = torch.sum(pdf * step.cpu())
        except ValueError:
            prob[i] = 0
    return prob.cpu()


def get_dists(args):
    cfg = Config.fromfile(args.val_cfg)
    param_save_location = os.path.join(cfg.model.roi_head.energy_save_path, 'weibull_dists.pkl')
    if args.use_fit and os.path.isfile(param_save_location):
        unk_params, known_params = torch.load(param_save_location)
    else:
        unk_params, known_params = analyse_energy(cfg)
    clear_energy_dir(cfg)
    unk_dist = create_distribution(unk_params['scale_unk'], unk_params['shape_unk'], unk_params['shift_unk'],)
    known_dist = create_distribution(known_params['scale_known'], known_params['shape_known'], known_params['shift_known'], )
    return unk_dist, known_dist


def clear_energy_dir(cfg):
    for filename in os.listdir(cfg.model.roi_head.energy_save_path):
        if filename == 'weibull_dists.pkl':
            continue
        file_path = os.path.join(cfg.model.roi_head.energy_save_path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)


def analyse_energy(cfg):
    files = os.listdir(cfg.model.roi_head.energy_save_path)
    temp = cfg.temperature
    unk = []
    known = []

    for id, file in enumerate(files):
        path = os.path.join(cfg.model.roi_head.energy_save_path, file)
        try:
            logits, classes = torch.load(path)
        except:
            continue
        num_seen_classes = cfg.model.roi_head.num_classes - 1
        lse = temp * torch.logsumexp(logits[:, :num_seen_classes] / temp, dim=1)
        # lse = torch.logsumexp(logits[:, :-2], dim=1)

        for i, cls in enumerate(classes):
            if cls == cfg.model.roi_head.num_classes or lse[i] < 0:
                continue
            if cls == cfg.model.roi_head.num_classes - 1:
                unk.append(lse[i].detach().cpu().tolist())
            else:
                known.append(lse[i].detach().cpu().tolist())

        if id % 100 == 0:
            pass
        # if id == 10:
        #     break

    wb_dist_param = []

    wb_unk = Fit_Weibull_3P(failures=unk, show_probability_plot=False, print_results=False)

    wb_dist_param.append({"scale_unk": wb_unk.alpha, "shape_unk": wb_unk.beta, "shift_unk": wb_unk.gamma})

    wb_known = Fit_Weibull_3P(failures=known, show_probability_plot=False, print_results=False)

    wb_dist_param.append(
        {"scale_known": wb_known.alpha, "shape_known": wb_known.beta, "shift_known": wb_known.gamma})

    param_save_location = os.path.join(cfg.model.roi_head.energy_save_path, 'weibull_dists.pkl')
    torch.save(wb_dist_param, param_save_location)
    return wb_dist_param


def append_uncertainty(bboxes, dists):
    bbox_counter = 0
    for i in tqdm.tqdm(range(len(bboxes))):
        for j in range(len(bboxes[i])):
            num_bboxes = bboxes[i][j].shape[0]
            if num_bboxes:
                uncertainty = compute_prob(bboxes[i][j][:, 5], dists[1]) #  - compute_prob(lse, dists[0])
                bboxes[i][j][:, 5] = uncertainty
                bbox_counter += num_bboxes
    return bboxes


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute metrics')
    parser.add_argument('val_cfg', help='val config')
    parser.add_argument('test_data', help='raw test data')
    parser.add_argument('-use-fit', action='store_true', help='use fit dists')
    args = parser.parse_args()
    return args

def load_outputs(outputs_file):
    import os
    path = os.path.dirname(outputs_file)
    all_files = [os.path.join(path, f) for f in os.listdir(path) if
                 os.path.isfile(os.path.join(path, f))]
    patched = []
    for file in sorted(all_files):
        if file.find(os.path.basename(outputs_file.split('.')[0])) >= 0 and file.find('chunk') >=0:
            chunk = mmcv.load(file)
            patched.extend(chunk)
    return patched if patched else mmcv.load(outputs_file)

def main():
    args = parse_args()
    dists = get_dists(args)
    outputs = load_outputs(args.test_data)
    bboxes = append_uncertainty(outputs, dists)
    save_path = args.test_data.split('.')[0] + '_uncertainty.pkl'
    mmcv.dump(bboxes, save_path)


if __name__ == '__main__':
    main()
