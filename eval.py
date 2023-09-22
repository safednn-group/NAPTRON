import argparse
import os
from collections import OrderedDict

import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute metrics')
    parser.add_argument('outputs', help='model outputs')
    parser.add_argument('config', help='test config')
    parser.add_argument('train_config', help='train config')
    args = parser.parse_args()
    return args


def prepare_label2cat(dataset, train_dataset):
    train_cats = train_dataset.get_cats()
    label2cat = {}
    for cat in train_cats:
        new_id = dataset.coco.get_cat_ids(cat_names=cat['name'])
        label2cat[train_dataset.cat2label[train_dataset.coco.get_cat_ids(cat_names=cat['name'])[0]]] = new_id[0]
    dataset.KNOWN_CLASSES = train_dataset.CLASSES
    return label2cat


def load_previous_results(cfg):
    results_dir = 'results/benchmark'
    filename = cfg.filename.split('/')[-1].split('.')[0] + '.pkl'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_path = os.path.join(results_dir, filename)
    if os.path.exists(results_path):
        results = mmcv.load(results_path)
    else:
        results = OrderedDict()
    return results


def dump_results(results, cfg):
    results_dir = 'results/benchmark'
    filename = cfg.filename.split('/')[-1].split('.')[0] + '.pkl'
    results_path = os.path.join(results_dir, filename)
    mmcv.dump(results, results_path)


def load_outputs(outputs_file):
    path = os.path.dirname(outputs_file)
    all_files = [os.path.join(path, f) for f in os.listdir(path) if
                 os.path.isfile(os.path.join(path, f))]
    patched = []
    for file in sorted(all_files):
        if file.find(os.path.basename(outputs_file.split('.')[0])) >= 0 and file.find('chunk') >= 0:
            chunk = mmcv.load(file)
            patched.extend(chunk)
    return patched if patched else mmcv.load(outputs_file)


def main():
    args = parse_args()
    outputs = load_outputs(args.outputs)
    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.test)
    train_cfg = Config.fromfile(args.train_config)

    train_cfg.data.test.type = 'CocoOODDataset'
    train_dataset = build_dataset(train_cfg.data.test)
    dataset.label2cat = prepare_label2cat(dataset, train_dataset)

    if 'unknown' in train_cfg.CLASSES:
        dataset.label2cat[max(dataset.label2cat.keys()) + 1] = 0
    results = load_previous_results(cfg)

    if 'rcnn' in args.config:
        nms_threshold = cfg.model.test_cfg.rcnn.score_thr
    else:
        nms_threshold = cfg.model.test_cfg.score_thr

    results[str(nms_threshold)] = dataset.evaluate_ood(outputs, metric=['uncertainty'], dump_certainties=True)

    dump_results(results, cfg)

    rename_certainties(args)


def rename_certainties(args):
    certainties = mmcv.load('certainties.pkl')
    save_path = args.outputs.split('.')[0] + '_certainties.pkl'
    mmcv.dump(certainties, save_path)


if __name__ == '__main__':
    main()
