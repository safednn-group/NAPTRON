import pathlib

benchmark_configs = pathlib.Path(__file__).parent.parent.parent / "config" / "benchmark"
template = "#!/bin/bash\n\nif ! ls work_dirs/outputs_dump/%s_model_outputs*_certainties.pkl 1> /dev/null 2>&1; then" \
           "\n\necho Evaluating algorithm from config file: %s; python eval.py work_dirs/outputs_dump/%s %s %s; fi"

if __name__ == "__main__":
    bdd_config_dir = benchmark_configs / "bdd"
    bdd_configs = list(bdd_config_dir.iterdir())
    voc2coco_config_dir = benchmark_configs / "voc2coco"
    voc2coco_configs = list(voc2coco_config_dir.iterdir())
    configs = voc2coco_configs + bdd_configs
    omitted_ood_methods = ["naptron", "owod", "gmm"]
    trained_methods = {"gaussian", "gmm", "oln", "opendet", "owod", "vos"}
    config_dir_map = {
        "voc2coco": "voc0712_cocofmt",
        "bdd": "bdd"
    }
    for config in configs:
        if any(m in config.name for m in omitted_ood_methods):
            continue
        method = f"_{config.name.split('_')[-2]}" if any(m in config.name for m in trained_methods) else ""
        cfg_dir = config_dir_map[config.parent.name] if method[1:] not in trained_methods else method[1:]
        detector = '_'.join(config.name.split('_')[:-2])
        if "bdd" in config.name:
            base_detector_config = f"config/{cfg_dir}/{detector}_r50_fpn_bddhalf{method}.py"
        elif "voc2coco" in config.name:
            base_detector_config = f"config/{cfg_dir}/{detector}_r50_fpn_voc0712_cocofmt{method}.py"
        with open(pathlib.Path(__file__).parent / (config.name.split(".")[0] + ".sh"), "w") as f:
            f.write(template % (config.name.split(".")[0], str(config), config.name.split(".")[0] + "_model_outputs.pkl", str(config), base_detector_config))