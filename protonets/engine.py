from tqdm import tqdm

class Engine(object):
    def __init__(self):
        hook_names = ['on_start', 'on_start_epoch', 'on_sample', 'on_forward',
                      'on_backward', 'on_end_epoch', 'on_update', 'on_end']

        self.hooks = { }  # Basically parts to execute upon different stages
        for hook_name in hook_names:
            self.hooks[hook_name] = lambda state: None  # Lambda variable because it takes in a function

    def train(self, **kwargs):
        state = {
            'model': kwargs['model'],
            'loader': kwargs['loader'],
            'optim_method': kwargs['optim_method'],
            'optim_config': kwargs['optim_config'],
            'max_epoch': kwargs['max_epoch'],
            'epoch': 0, # epochs done so far
            't': 0, # samples seen so far  ---- when t % 100 == 0, calculate NC metrics in between
            'batch': 0, # samples seen in current epoch
            'stop': False
        }

        # Initializes Adam optimizer with parameters, lr, wd
        state['optimizer'] = state['optim_method'](state['model'].parameters(), **state['optim_config'])

        self.hooks['on_start'](state)
        while state['epoch'] < state['max_epoch'] and not state['stop']:
            state['model'].train()  # Puts the model into training mode -> for BatchNorm

            self.hooks['on_start_epoch'](state)

            state['epoch_size'] = len(state['loader'])

            for sample in tqdm(state['loader'], desc="Epoch {:d} train".format(state['epoch'] + 1)):
                state['sample'] = sample
                self.hooks['on_sample'](state)

                state['optimizer'].zero_grad()
                loss, state['output'] = state['model'].loss(state['sample'])
                self.hooks['on_forward'](state)

                loss.backward()  # Calculate gradients
                self.hooks['on_backward'](state)

                state['optimizer'].step()  # Update parameters using Adam with new learning rate

                state['t'] += 1
                state['batch'] += 1
                self.hooks['on_update'](state)

            # Calculate neural collapse metrics after each epoch (on_end_epoch) using model parameters to calculate NC metrics

            state['epoch'] += 1
            state['batch'] = 0  # Equals batch-size before resetting (number of samples seen in one epoch)
            self.hooks['on_end_epoch'](state)

        self.hooks['on_end'](state)
