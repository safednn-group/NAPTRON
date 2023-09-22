import pathlib

benchmark_configs = pathlib.Path(__file__).parent.parent.parent / "config" / "benchmark"
template = "#!/bin/bash\n\n if ! ls work_dirs/outputs_dump/%s_model_outputs*.pkl 1> /dev/null 2>&1; then" \
           "\n\npython safednn_naptron/utils/test.py %s %s --eval bbox %s; fi"

if __name__ == "__main__":
    bdd_config_dir = benchmark_configs / "bdd"
    bdd_configs = list(bdd_config_dir.iterdir())
    voc2coco_config_dir = benchmark_configs / "voc2coco"
    voc2coco_configs = list(voc2coco_config_dir.iterdir())
    configs = voc2coco_configs + bdd_configs
    trained_methods = {"gaussian", "gmm", "oln", "opendet", "owod", "vos"}
    for config in configs:
        method = f"_{config.name.split('_')[-2]}" if any(m in config.name for m in trained_methods) else ""
        detector = '_'.join(config.name.split('_')[:-2])
        progress_flag = "--override-batch-size" if "naptron" in config.name else ""
        if "bdd" in config.name:
            detector_checkpoint = f"work_dirs/{detector}_r50_fpn_bddhalf{method}/latest.pth"
        elif "voc2coco" in config.name:
            detector_checkpoint = f"work_dirs/{detector}_r50_fpn_voc0712_cocofmt{method}/latest.pth"
        with open(pathlib.Path(__file__).parent / (config.name.split(".")[0] + ".sh"), "w") as f:
            f.write(template % (config.name.split(".")[0], str(config), detector_checkpoint, progress_flag))