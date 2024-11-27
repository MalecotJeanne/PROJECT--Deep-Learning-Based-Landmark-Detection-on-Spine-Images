from models.hrnet import HRNet
from models.unet import UNet

models = {
    "hrnet": HRNet,
    "unet": UNet,
}

def init_model(name, *args, **kwargs):
    if name not in models.keys():
        raise KeyError("Unknown models: {}".format(name)) #TODO: Add a logger here
    return models[name](*args, **kwargs)
