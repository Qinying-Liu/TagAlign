from mmcv.utils import Registry
from omegaconf import OmegaConf

MODELS = Registry("model")


def build_model(config):
    model = MODELS.build(OmegaConf.to_container(config, resolve=True))
    return model
