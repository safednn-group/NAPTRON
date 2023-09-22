import argparse
import numpy as np
import torch
import mmcv
from tqdm import tqdm
from mmdet.datasets import build_dataset
from mmcv.utils.misc import import_modules_from_strings

from monitor import FullNetMonitor


def parse_args_nap():
    parser = argparse.ArgumentParser(
        description='NAP')
    parser.add_argument('train', help='train results path')
    parser.add_argument('train_config', help='train config')
    parser.add_argument('test', help='test results path')
    parser.add_argument('test_config', help='test config', default='', nargs="?")
    parser.add_argument('layer_id', help='index of the monitored layer')
    parser.add_argument('-mean', action='store_true',
                        help='mean hamming distance from known patterns if true; else the nearest')
    args = parser.parse_args()
    return args


def count_classes_labels(labels):
    count_class = dict()
    for c in range(int(labels.max() + 1)):
        count_class[c] = (labels == c).sum()
    return count_class, labels


def append_zeros(bboxes, activations):
    for i in range(len(bboxes)):
        for j in range(len(bboxes[i])):
            num_bboxes = bboxes[i][j].shape[0]
            if num_bboxes:
                zeroes = np.zeros((num_bboxes, activations[i].shape[-1]))
                bboxes[i][j] = np.column_stack((bboxes[i][j], zeroes))
    return bboxes


def append_uncertainty(bboxes, uncertainty, labels):
    split_sizes = [labels[j].shape[0] for j in range(len(bboxes))]
    uncertainty = torch.tensor(uncertainty).split(split_sizes)
    uncertainty_bbox_fmt = [[uncertainty[j][labels[j] == i] for i in range(len(bboxes[0]))] for j in range(len(bboxes))]
    for i in range(len(bboxes)):
        bboxes_per_img = 0
        for j in range(len(bboxes[i])):
            num_bboxes = bboxes[i][j].shape[0]
            if num_bboxes:
                uncertainty_img = uncertainty_bbox_fmt[i][j]
                bboxes[i][j] = np.column_stack((bboxes[i][j], uncertainty_img))
                bboxes_per_img += num_bboxes
    return bboxes


def get_tp_activations(certainties, activations, bboxes, labels, iou=0, score_thr=0.):
    activations_bbox_fmt = [[activations[j][labels[j] == i] for i in range(len(bboxes[0]))] for j in range(len(bboxes))]
    mask = (certainties['tp_logit_certainty'][iou][0] > score_thr)
    train_tp_activations = torch.zeros((len(certainties['tp_img_id'][iou][0][mask]), activations[0].shape[-1]),
                                       device='cuda')
    train_tp_labels = np.zeros((len(certainties['tp_img_id'][iou][0][mask])))

    for i, img_id in tqdm(enumerate(certainties['tp_img_id'][iou][0][mask]), total=train_tp_labels.shape[0]):
        bbox_id = int(certainties['tp_bbox_id'][iou][0][mask][i])
        lbl = int(certainties['tp_bbox_label'][iou][0][mask][i])

        train_tp_activations[i] = activations_bbox_fmt[int(img_id)][lbl][bbox_id]
        train_tp_labels[i] = lbl

    return train_tp_activations, train_tp_labels


def prepare_nap_monitor(args):
    train = mmcv.load(args.train)
    train_bboxes, train_activations, shapes, train_labels = parse_nap_detector_outputs(train)
    class_num = len(train_bboxes[0])

    train_cfg = mmcv.Config.fromfile(args.train_config)
    custom_imports = dict(
        imports=['safednn.uncertainty.coco_eval_ood',
                 'safednn.uncertainty.coco_ood_dataset'],
        allow_failed_imports=False)
    import_modules_from_strings(**custom_imports)
    train_cfg.data.test.type = 'CocoOODDataset'
    train_dataset = build_dataset(train_cfg.data.test)
    train_bboxes = append_zeros(train_bboxes, train_activations)
    try:
        train_dataset.evaluate_ood(train_bboxes, metric=['uncertainty'], dump_certainties=True)
    except ValueError as e:
        print(e)
        print("Continuing")

    train_certainties = mmcv.load('certainties.pkl')
    save_path = args.train.split('.')[0] + '_certainties.pkl'
    mmcv.dump(train_certainties, save_path)

    iou_id = 0  # iou = 0.5 + 0.05 * iou_id
    train_score_thr = 0.

    train_tp_activations, train_tp_labels = get_tp_activations(train_certainties, train_activations,
                                                               train_bboxes, train_labels, iou=iou_id,
                                                               score_thr=train_score_thr)

    save_path = args.train.split('.')[0] + '_train_tp_activations.pkl'
    mmcv.dump(train_tp_activations, save_path)

    examples_per_class, _ = count_classes_labels(train_tp_labels)

    train_tp_activations, monitored_layers_shapes = prepare_patterns(train_tp_activations, shapes, int(args.layer_id))

    monitor = FullNetMonitor(class_num, train_tp_activations.device,
                             layers_shapes=monitored_layers_shapes)
    monitor.set_class_patterns_count(examples_per_class)
    monitor.add_neuron_pattern(train_tp_activations, train_tp_labels)
    monitor.cut_duplicates()

    return monitor


def prepare_patterns(patterns, shapes, chosen_layer_id=None):
    layers, layers_shapes = split_layerwise(patterns, shapes)
    if chosen_layer_id:
        layers = [layers[chosen_layer_id]]
        layers_shapes = [layers_shapes[chosen_layer_id]]
    prepared_pattern = binarize_patterns(layers)
    return prepared_pattern, layers_shapes


def split_layerwise(patterns, shapes):
    layers = []
    layers_shapes = []
    prev = 0
    for shape in shapes:
        layers.append(patterns[:, prev: prev + shape])
        layers_shapes.append(shape)
        prev += shape
    return layers, layers_shapes


def binarize_patterns(layers, q=0.):
    zero_tensor = torch.zeros(1, device=layers[0].device)
    bin_layers = []
    for layer in layers:
        # bin_layers.append(
        #     torch.where(torch.abs(layer) > torch.quantile(torch.abs(layer), q, dim=1).unsqueeze(1), layer, zero_tensor))
        bin_layers.append((layer > 0).type(torch.uint8))

    concat_pattern = torch.cat(bin_layers, dim=1)
    return concat_pattern


def parse_nap_detector_outputs(outputs):
    bboxes, activations, shapes, labels = outputs[::4], outputs[1::4], outputs[2::4], outputs[3::4]
    if not isinstance(activations[0], torch.Tensor):
        activations = [i[0] for i in activations]
    if len(bboxes[0]) == 1:
        bboxes = [i[0] for i in bboxes]
    labels = [i.cpu() if isinstance(i, torch.Tensor) else i[0].cpu() for i in labels]
    shape = shapes[0]
    return bboxes, activations, shape, labels


def load_outputs(outputs_file):
    import os
    path = os.path.dirname(outputs_file)
    all_files = [os.path.join(path, f) for f in os.listdir(path) if
                 os.path.isfile(os.path.join(path, f))]
    patched = []
    for file in sorted(all_files):
        if file.find(os.path.basename(outputs_file.split('.')[0])) >= 0 and file.find('chunk') >= 0:
            chunk = mmcv.load(file)
            patched.extend(chunk)
    return patched if patched else mmcv.load(outputs_file)


if __name__ == '__main__':
    args = parse_args_nap()
    monitor = prepare_nap_monitor(args)
    # load test patterns and bboxes
    test_results = load_outputs(args.test)
    test_bboxes, test_activations, shapes, test_labels = parse_nap_detector_outputs(test_results)
    test_activations_tensor = torch.cat(test_activations).cuda()
    test_labels_np = torch.cat(test_labels).numpy()

    test_activations_tensor, _ = prepare_patterns(test_activations_tensor, shapes, int(args.layer_id))

    test_distances = monitor.compute_hamming_distance(test_activations_tensor, test_labels_np, mean=args.mean)
    # test_distances = test_distances[:, int(args.layer_id)]  # last layer only

    test_outputs_uncertainty = append_uncertainty(test_bboxes, test_distances,
                                                  test_labels)  # NAP uncertainty is appended to every bbox

    save_path = args.test.split('.')[0] + '_uncertainty.pkl'
    mmcv.dump(test_outputs_uncertainty, save_path)
