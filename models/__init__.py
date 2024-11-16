from models.hrnet import HRNet

models = {
    "hrnet": HRNet,
}

def init_model(name, *args, **kwargs):
    if name not in models.keys():
        raise KeyError("Unknown models: {}".format(name)) #TODO: Add a logger here
    return models[name](*args, **kwargs)
