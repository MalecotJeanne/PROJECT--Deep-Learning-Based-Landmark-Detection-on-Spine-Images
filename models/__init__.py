from models.hrnet import HRNet
from models.unet import UNet
from models.unet_base import UNet_base
from models.dynunet import DynUNet

models = {
    "hrnet": HRNet,
    "unet": UNet,
    "unet_base": UNet_base,
    "dynunet": DynUNet,
}

def init_model(name, *args, **kwargs):
    if name not in models.keys():
        raise KeyError("Unknown models: {}".format(name)) 
    return models[name](*args, **kwargs)
