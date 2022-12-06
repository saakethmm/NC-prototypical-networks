import torch
import torch.nn.functional as F


# Within-class covariance matrix
def compute_sigma_w(device, features, means):

    num_data = 0
    Sigma_W = 0

    for target in before_class_dict.keys():
        for features in class_feature_list:
            features = torch.from_numpy(features).to(device)
            Sigma_W_batch = (features - mu_c_dict[target].unsqueeze(0)).unsqueeze(2) * (features - mu_c_dict[target].unsqueeze(0)).unsqueeze(1)
            Sigma_W += torch.sum(Sigma_W_batch, dim=0)
            num_data += features.shape[0]

    # Average over all the data, equivalent to averaging over all examples in each class
    Sigma_W /= num_data
    return Sigma_W.detach().cpu().numpy()


# Between-class covariance matrix
def compute_sigma_b(mu_c_dict, mu_G):
    Sigma_B = 0
    # Number of classes
    K = len(mu_c_dict)
    for i in range(K):
        Sigma_B += (mu_c_dict[i] - mu_G).unsqueeze(1) @ (mu_c_dict[i] - mu_G).unsqueeze(0)

    # Averaging over all classes
    Sigma_B /= K

    return Sigma_B.detach().cpu().numpy()

# NC2: Same as above but instead with the last-layer features themselves
def compute_ETF_feature(mu_c_dict, mu_G):
    """
    args:
    @ mu_c_dict: dictionary of class feature mean
    @ mu_G: Global mean of features
    Both of the above parameter could be obtained from the compute_info function
    """
    device = mu_G.device
    classes = list(mu_c_dict.keys())
    K = len(classes)
    fea_len = mu_c_dict[classes[0]].shape[0]

    H_bar = torch.zeros(K, fea_len).to(device)
    for i, k in enumerate(mu_c_dict):
        H_bar[i] = mu_c_dict[k] - mu_G  # Subtract global mean from class mean

    # First normalize the class feature deviations to unit norm, then Frobenius norm to 1
    # Intuition: If one vector had generally large deviations compared to another, the other vector
    #  would be more harshly punished in Frobenius norm (whereas normalizing first would induce same
    #  effect across all of them)
    H_bar = F.normalize(H_bar, dim=1)

    HHT = torch.mm(H_bar, H_bar.T)
    HHT /= torch.norm(HHT, p='fro')

    sub = (torch.eye(K) - 1 / K * torch.ones((K, K))).to(device) / pow(K - 1, 0.5)

    ETF_metric_tilde = torch.norm(HHT - sub, p='fro')
    return ETF_metric_tilde.detach().cpu().numpy().item()