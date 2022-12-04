MODEL_REGISTRY = {}

def register_model(model_name):
    def decorator(f):
        MODEL_REGISTRY[model_name] = f
        return f

    return decorator

# Benefit of doing it this way is you can register several models into a dictionary and use whichever one you want
def get_model(model_name, model_opt):
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name](**model_opt)  # Calls load_protonet_conv with model parameters in run_train
    else:
        raise ValueError("Unknown model {:s}".format(model_name))
