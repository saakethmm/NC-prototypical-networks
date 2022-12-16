from tqdm import tqdm

from protonets.utils import filter_opt
from protonets.models import get_model

def load(opt):
    model_opt = filter_opt(opt, 'model')  # Filter out arguments for model details (model_name, x_dim, hid_dim, z_dim)
    model_name = model_opt['model_name']  # Get model name - default is protonet_conv

    del model_opt['model_name']

    return get_model(model_name, model_opt)

def evaluate(model, data_loader, meters, desc=None):
    model.eval()

    for field, meter in meters.items():
        meter.reset()

    if desc is not None:
        data_loader = tqdm(data_loader, desc=desc)

    for sample in data_loader:
        _, output = model.loss(sample)
        for field, meter in meters.items():
            meter.add(output[field])

    return meters
