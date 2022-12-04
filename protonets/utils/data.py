import protonets.data

def load(opt, splits):
    if opt['data.dataset'] == 'omniglot':
        ds = protonets.data.omniglot.load(opt, splits)  # Returns dataloader given the splits
    # elif: ... This is where we should add functionality for miniImageNet, CIFAR-FS, etc.
    else:
        raise ValueError("Unknown dataset: {:s}".format(opt['data.dataset']))

    return ds
