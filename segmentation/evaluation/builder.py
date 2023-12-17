import mmcv
import yaml
import torch
from omegaconf import OmegaConf
from datasets import get_template

from .tagalign_seg import TagAlignSegInference


def build_dataset_class_tokens(text_transform, template_set, classnames):
    tokens = []
    templates = get_template(template_set)
    for classname in classnames:
        tokens.append(
            torch.stack([text_transform(template.format(classname)) for template in templates])
        )
    # [N, T, L], N: number of instance, T: number of captions (including ensembled), L: sequence length
    tokens = torch.stack(tokens)

    return tokens


def build_seg_dataset(config, key):
    """Build a dataset from config."""
    cfg = mmcv.Config.fromfile(config)
    if 'ImageNet' in key:
        from mmsegmentation_obsolete.mmseg.datasets import build_dataset
    else:
        from mmseg.datasets import build_dataset

    dataset = build_dataset(cfg.data.test)
    return dataset


def build_seg_dataloader(dataset, key):
    if 'ImageNet' in key:
        from mmsegmentation_obsolete.mmseg.datasets import build_dataloader
    else:
        from mmseg.datasets import build_dataloader
    # batch size is set to 1 to handle varying image size (due to different aspect ratio)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=True,
        shuffle=False,
        persistent_workers=True,
        pin_memory=False,
    )
    return data_loader


def build_seg_inference(
    model,
    dataset,
    text_transform,
    config,
    seg_config,
):
    dset_cfg = mmcv.Config.fromfile(seg_config)  # dataset config
    with_bg = dataset.CLASSES[0] == "background"
    if with_bg:
        classnames = dataset.CLASSES[1:]
    else:
        classnames = dataset.CLASSES
    text_tokens = build_dataset_class_tokens(text_transform, config.evaluate.template, classnames)
    text_embedding = model.build_text_embedding(text_tokens)
    kwargs = dict(with_bg=with_bg)
    if hasattr(dset_cfg, "test_cfg"):
        kwargs["test_cfg"] = dset_cfg.test_cfg

    model_type = config.model.type
    if model_type == "TagAlign":
        seg_model = TagAlignSegInference(model, text_embedding, **kwargs, **config.evaluate)
    else:
        raise ValueError(model_type)

    seg_model.CLASSES = dataset.CLASSES
    seg_model.PALETTE = dataset.PALETTE

    return seg_model
