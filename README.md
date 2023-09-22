## Paper info

This repository is the official implementation of [Detecting Out-of-distribution Objects Using Neuron Activation Patterns](https://arxiv.org/abs/2307.16433)

Paper has been accepted to 26th European Conference on Artificial Intelligence ECAI 2023.


```
@misc{olber2023detecting,
      title={Detecting Out-of-distribution Objects Using Neuron Activation Patterns}, 
      author={Bartłomiej Olber and Krystian Radlak and Krystian Chachuła and Jakub Łyskawa and Piotr Frątczak},
      year={2023},
      eprint={2307.16433},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## OOD for object detection algorithms comparison 

This repository contains code and automation scripts used for generating results described in the Table 4 of the accompanying paper.

### Datasets preparation
Download datasets into `data` directory

- Download BDD100k dataset and convert it to COCO format 
    - Possibly you would have to use the [official distribution](https://bdd-data.berkeley.edu/portal.html#download) 
    - If so, download images by clicking `100k Images` button and `Detection 2020 Labels` for annotations.
    - Use official [toolkit](https://github.com/bdd100k/bdd100k) to convert annotation to COCO format - official [docs](https://doc.bdd100k.com/format.html)
- Download Pascal VOC 2007-2012 dataset and convert it to COCO format or use this `bash scripts/download_voc.sh data`
- Download COCO dataset - `bash scripts/download_coco.sh data`

### Experiment preparation

- Prepare datasets according to this [instruction](./readme_datasets.md)
- Prepare configs according to this [instruction](./config/_base_/datasets/readme.md)
- Make sure you've got safednn_naptron library installed. See [instruction](./install_safednn.md)
- Generate test scripts `python scripts/generate_bboxes/generate_scripts.py`
- Generate eval scripts `python scripts/eval/generate_eval_scripts.py`

### Train detectors

Train all used in the comparison detectors on BDD100k and Pascal VOC datasets:

```
bash scripts/train_all.sh
```

State dicts of the trained detectors should be saved in an appropriate subdirectory of `work_dirs` as `latest.pth`.

### Generate test detections 

Generate detections for two validation datasets - BDD100k -> BDD100k, Pascal VOC -> COCO

```
bash scripts/generate_bboxes/generate_bboxes_all_methods.sh
```

### Evaluation

Apply postprocessing logic to detections if needed (depends on a method) and evaluate

```
bash scripts/eval/eval_all.sh
```

### Gather results

Gather and plot OOD detection AUROC and FPR@95TPR results 

```
python gather.py
```

### Memory issues
NAPTRON outputs generated for large datasets are gathered in memory during inference and then 
dumped all at once. In case you have no sufficient RAM to dump many GB to the hard drive,
try to set chunking by adding `chunks_count=N` in the config file that causes problem.
See example in: `config/benchmark/voc2coco/fcos_naptron_voc2coco.py`
```
output_handler = dict(
    type="simple_dump",
    chunks_count=5
)
```


