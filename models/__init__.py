from models.hrnet import HRNet
from models.unet import UNet
from models.retina import RetinaNet
from models.unet_base import UNet_base

models = {
    "hrnet": HRNet,
    "unet": UNet,
    "retina": RetinaNet,
    "unet_base": UNet_base
}

def init_model(name, *args, **kwargs):
    if name not in models.keys():
        raise KeyError("Unknown models: {}".format(name)) 
    return models[name](*args, **kwargs)
