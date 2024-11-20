import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from scipy.stats import wasserstein_distance

# Define a new class called gap_reg that inherits from torch.nn.Module

class wasserstein_reg(torch.nn.Module):
    """
    As implemented by Shalit et al., translated to PyTorch: https://github.com/clinicalml/cfrnet/blob/master/cfr/util.py
    """
    #! fix device issues, now dirty fix put everything on CPU
    def __init__(self, mode="dp", local_reg=True, threshold_based=True):
        super(wasserstein_reg, self).__init__()
        self.mode = mode
        self.local_reg = local_reg
        # Set whether regularization happens between two cut-off values (True) or two percentiles (False)
        self.threshold_based = threshold_based

    def forward(self, y_pred, s, y_gt, pct_a=0.0, pct_b=1.0):
        #Default settings of Shalit et al.:
        lam = 10
        its = 10
        sq = False
        backpropT = False

        y0 = y_pred[s == 0].cpu()
        y1 = y_pred[s == 1].cpu()

        nc = float(y0.shape[0])
        nt = float(y1.shape[0])

        len_y0 = len(y0)
        len_y1 = len(y1)

        # Determine the length of the larger tensor
        max_len = max(len_y0, len_y1)

        y0_shape = y0.shape

        #TODO: For local regularization, check whether this is best/cleanest way to implement
        if self.local_reg:
            # Sort probabilities in ascending order
            sorted_y0, _ = torch.sort(y0)
            sorted_y1, _ = torch.sort(y1)

            if self.threshold_based:
                #only select values between 0.7 & 1
                y0 = sorted_y0[(sorted_y0 >= pct_a) & (sorted_y0 <= pct_b)]
                y1 = sorted_y1[(sorted_y1 >= pct_a) & (sorted_y1 <= pct_b)]
                if len(y0) == 0 or len(y1) == 0:
                    raise ValueError(f"At least one group does not have predictions in [{pct_a},  {pct_b}]. "
                                     f"Impossible to regularize with [threshold_based==True]")
            else:

                index_a_0 = int(pct_a * len_y0)
                index_b_0 = int(pct_b * len_y0)
                index_a_1 = int(pct_a * len_y1)
                index_b_1 = int(pct_b * len_y1)

                y0 = sorted_y0[index_a_0:index_b_0]
                y1 = sorted_y1[index_a_1:index_b_1]

        # Compute distance matrix
        #M = torch.sqrt(torch.cdist(y1, y0, p=2) ** 2)
        M = torch.sqrt(torch.cdist(y1.unsqueeze(1), y0.unsqueeze(1), p=2) ** 2) #cdist requires at lest 2D tensor (and received 1D)

        # Estimate lambda and delta
        M_mean = torch.mean(M)
        M_drop = F.dropout(M, p=0.5)  # You can adjust the dropout rate if needed
        delta = torch.max(M).detach()  # Detach to prevent gradients from flowing
        eff_lam = (lam / M_mean).detach()  # Detach to prevent gradients from flowing

        # Compute new distance matrix with additional rows and columns
        row = delta * torch.ones((1, M.shape[1]), device=M.device)
        col = torch.cat((delta * torch.ones((M.shape[0], 1), device=M.device), torch.zeros((1, 1), device=M.device)),
                        dim=0)
        Mt = torch.cat((M, row), dim=0)
        Mt = torch.cat((Mt, col), dim=1)

        # Compute marginal vectors for treated and control groups
        p = 0.5 #In original code: given as parameter. Now just fixed on 0.5
#       
        a_indices = torch.where(s > 0)[0]

        a = torch.cat([(p * torch.ones(len(y1)) / nt).unsqueeze(1), (1 - p) * torch.ones((1, 1))], dim=0)

        b_indices = torch.where(s < 1)[0]

        b = torch.cat([((1 - p) * torch.ones(len(y0)) / nc).unsqueeze(1), p * torch.ones((1, 1))], dim=0)

        # Compute kernel matrix and related matrices
        Mlam = eff_lam * Mt
        K = torch.exp(-Mlam) + 1e-6  # Added constant to avoid nan
        U = K * Mt
        ainvK = K / a

        # Compute u matrix iteratively
        u = a
        for i in range(its):
            u = 1.0 / torch.matmul(ainvK, (b / torch.t(torch.matmul(torch.t(u), K))))

        # Compute v matrix
        v = b / torch.t(torch.matmul(torch.t(u), K))

        # Compute transportation matrix T
        T = u * (torch.t(v) * K)

        if not backpropT:
            T = T.detach()  # Detach T if backpropagation is not needed

        # Compute E matrix and final Wasserstein distance D
        E = T * Mt
        D = 2 * torch.sum(E)

        #define reg_loss
        reg_loss = D

        torchtensor = torch.Tensor([0])

        return reg_loss, torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])


class KL_divergence_reg(torch.nn.Module):
    def __init__(self, mode="dp", local_reg=True, threshold_based=True):
        super(KL_divergence_reg, self).__init__()
        self.mode = mode
        self.local_reg = local_reg
        # Set whether regularization happens between two cut-off values (True) or two percentiles (False)
        self.threshold_based = threshold_based

    def forward(self, y_pred, s, y_gt, pct_a=0.0, pct_b=1.0):
        y0 = y_pred[s == 0].cpu()
        y1 = y_pred[s == 1].cpu()

        len_y0 = len(y0)
        len_y1 = len(y1)

        # Sort probabilities in ascending order
        sorted_y0, _ = torch.sort(y0)
        sorted_y1, _ = torch.sort(y1)

        y0 = sorted_y0
        y1 = sorted_y1

        if self.local_reg:

            if self.threshold_based:
                # only select values between 0.7 & 1
                y0 = sorted_y0[(sorted_y0 >= pct_a) & (sorted_y0 <= pct_b)]
                y1 = sorted_y1[(sorted_y1 >= pct_a) & (sorted_y1 <= pct_b)]

                if len(y0) == 0 or len(y1) == 0:
                    raise ValueError(f"At least one group does not have predictions in [{pct_a},  {pct_b}]. "
                                     f"Impossible to regularize with [threshold_based==True]")

            else:
                # only select values between percentile 0.7 and 1 (per group)
                index_a_0 = int(pct_a*len_y0)
                index_b_0 = int(pct_b*len_y0)
                index_a_1 = int(pct_a*len_y1)
                index_b_1 = int(pct_b*len_y1)

                y0 = sorted_y0[index_a_0:index_b_0]
                y1 = sorted_y1[index_a_1:index_b_1]

        len_y0 = len(y0)
        len_y1 = len(y1)

        # Determine the length of the larger tensor
        max_len = max(len_y0, len_y1)

        # Interpolate or pad the smaller tensor to match the length of the larger tensor
        if len_y0 < max_len:
            y0 = F.interpolate(y0.unsqueeze(0).unsqueeze(0), size=max_len, mode='linear',
                                      align_corners=True).squeeze(0).squeeze(0)
        elif len_y1 < max_len:
            y1 = F.interpolate(y1.unsqueeze(0).unsqueeze(0), size=max_len, mode='linear',
                                      align_corners=True).squeeze(0).squeeze(0)

        reg_loss = F.kl_div(F.log_softmax(y0, dim=0), F.softmax(y1, dim=0), reduction='batchmean')

        #Multiply KL_div absolute value with 1M (otherwise lambda needs to take extreme values to see impact)
        reg_loss = reg_loss*1000000

        return reg_loss, torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])