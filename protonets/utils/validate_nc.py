import torch
import torch.nn.functional as F
import torch.linalg as linalg


# Within-class covariance matrix
def compute_sigma_w(features, means):
    """
    N = # of classes, K = # of examples for each class
    features: (N, K, z_dim)
    means: (N, z_dim)
    """
    N, K = features.shape(0), features.shape(1)

    diff = (features - means.unsqueeze(1)).unsqueeze(3)  # (N_support, K_support, z_dim, 1)
    diff_tr = diff.view(N, K, 1, -1)  # (N_support, K_support, 1, z_dim)

    sigma_w = 0
    for i in range(N):
        sigma_w += torch.bmm(diff[i], diff_tr[i])
    sigma_w = torch.sum(sigma_w, dim=0) / (N * K)

    return sigma_w.detach()

# Between-class covariance matrix
def compute_sigma_b(means, global_mean):
    """
    N = # of classes
    means: (N, z_dim)
    global_mean: (z_dim)
    """
    N = means.shape(0)

    diff = (means - global_mean.unsqueeze(0)).unsqueeze(2)  # (N, z_dim, 1)
    diff_tr = diff.view(N, 1, -1)  # (N, 1, z_dim)

    sigma_b = torch.bmm(diff, diff_tr).mean(dim=0)  # (z_dim, z_dim)

    return sigma_b.detach()


def compute_nc1(features, means):

    N = features.shape(0)
    global_mean = means.mean(0)

    sigma_w = compute_sigma_w(features, means)
    sigma_b = compute_sigma_b(means, global_mean)

    nc1_metric = 1 / N * torch.trace(torch.matmul(sigma_w, linalg.pinv(sigma_b)))

    return nc1_metric.detach()


def compute_nc2(means):
    """
    N = # of classes
    means: (N, z_dim)
    """
    N = means.shape[0]
    device = means.device

    global_mean = means.mean(0).unsqueeze(0)  # (z_dim)
    h_bar = means - global_mean  # (N, z_dim)

    # First normalize the class feature deviations to unit norm, then Frobenius norm to 1
    # Intuition: If one vector had generally large deviations compared to another, the other vector
    #  would be more harshly punished in Frobenius norm (whereas normalizing first would induce same
    #  effect across all of them)
    h_bar = F.normalize(h_bar, dim=1)

    hh_tr = torch.mm(h_bar, h_bar.T)
    hh_tr /= torch.norm(hh_tr, p='fro')  # (N, N)

    simplex_etf = (torch.eye(N) - 1 / N * torch.ones((N, N))).to(device) / torch.pow(N - 1, 0.5)  # (N, N)

    nc2_metric = torch.norm(hh_tr - simplex_etf, p='fro')

    return nc2_metric.detach()
