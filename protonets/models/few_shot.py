import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from protonets.models import register_model

from .utils import euclidean_dist
import protonets.utils.validate_nc as nc

from protonets.models.resnet import ResNet18, ResNet34, ResNet50

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Protonet(nn.Module):
    def __init__(self, encoder):
        super(Protonet, self).__init__()
        
        self.encoder = encoder

    def loss(self, sample, model_name):
        xs = Variable(sample['xs'])  # support, shape: (N_support, K_support, C_in, H, W)
        xq = Variable(sample['xq'])  # query, shape: (N_query, K_query, C_in, H, W)

        n_class = xs.size(0)  # N_support
        assert xq.size(0) == n_class  # N_support = N_query
        n_support = xs.size(1)  # K_support
        n_query = xq.size(1)  # K_query

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()

        # Shape: (N_support * K_support + N_support * K_query, C_in, H, W)
        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)

        if model_name != 'protonet_conv':
            x = x.repeat(1, 3, 1, 1)  # Clever solution to make BW image -> RGB

        z = self.encoder.forward(x)  # Flattened so shape is (N_support * K_support + N_support * K_query, 64)
        z_dim = z.size(-1)

        # Only consider first N_support * K_support images, viewed as (N_support, K_support, 64)
        zs_proto_mean = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)  # Not detached
        zq = z[n_class*n_support:]  # Not detached

        zs_proto = z[:n_class * n_support].view(n_class, n_support, z_dim).detach()
        zq_proto = zq.view(n_class, n_query, z_dim).detach()
        zq_proto_mean = zq_proto.mean(1)

        nc1_zs = nc.compute_nc1(zs_proto, zs_proto_mean.detach())
        nc2_zs = nc.compute_nc2(zs_proto_mean.detach())

        nc1_zq = nc.compute_nc1(zq_proto, zq_proto_mean)
        nc2_zq = nc.compute_nc2(zq_proto_mean)

        # One reason we use prototypical networks is that the feature vectors produced can be
        # easily built upon to find NC metrics
        dists = euclidean_dist(zq, zs_proto_mean)

        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item(),
            'nc1_support': nc1_zs.item(),
            'nc2_support': nc2_zs.item(),
            'nc1_query': nc1_zq.item(),
            'nc2_query': nc2_zq.item()
        }

@register_model('protonet_conv')
def load_protonet_conv(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']

    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    encoder = nn.Sequential(
        conv_block(x_dim[0], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, z_dim),
        Flatten()
    )

    return Protonet(encoder)  # Returns an initialized prototypical network to be trained

@register_model('resnet18')
def load_resnet18(**kwargs):
    return Protonet(ResNet18())

@register_model('resnet34')
def load_resnet34(**kwargs):
    return Protonet(ResNet34())