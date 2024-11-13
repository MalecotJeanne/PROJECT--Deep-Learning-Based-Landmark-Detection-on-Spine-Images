from models.hrnet import HRNet
from models.resnet import ResNet

models = {
    "hrnet": HRNet,
    "resnet": ResNet
}

def init_model(name, *args, **kwargs):
    if name not in models.keys():
        raise KeyError("Unknown models: {}".format(name)) #TODO: Add a logger here
    return models[name](*args, **kwargs)
