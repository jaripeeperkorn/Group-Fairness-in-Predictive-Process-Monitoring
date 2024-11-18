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

    def forward(self, y_pred, s, y_gt, pct_a, pct_b):
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

    def forward(self, y_pred, s, y_gt, pct_a, pct_b):
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





class dp_reg(torch.nn.Module):
    # Define the constructor for the class
    def __init__(self, mode = "dp", local_reg=False, threshold_based=True):
        # Call the constructor of the parent class
        super(dp_reg, self).__init__()
        # Set the mode attribute to the value passed in as an argument
        self.mode = mode
        # Set whether regularization happens between two percentiles (Boolean)
        self.local_reg = local_reg
        # Set whether regularization happens between two cut-off values (True) or two percentiles (False)
        self.threshold_based = threshold_based

    # Define the forward method for the class
    def forward(self, y_pred, s, y_gt, pct_a=0.0, pct_b=1.0):
        # Select the predicted values corresponding to s == 0
        y0 = y_pred[s == 0]
        # Select the predicted values corresponding to s == 1
        y1 = y_pred[s == 1]

        # Sort probabilities in ascending order
        sorted_y0, _ = torch.sort(y0)
        sorted_y1, _ = torch.sort(y1)

        len_y0 = len(y0)
        len_y1 = len(y1)

        # Calculate the regularization loss as the absolute difference between the means of y0 and y1
        if self.local_reg:
            if self.threshold_based:
                # Check if either of the filtered arrays is empty
                if len(sorted_y0[(sorted_y0 >= pct_a) & (sorted_y0 <= pct_b)]) == 0 or len(sorted_y1[(sorted_y0 >= pct_a) & (sorted_y0 <= pct_b)]) == 0:
                    raise ValueError(f"At least one group does not have predictions predictions in [{pct_a},  {pct_b}]. Impossible to regularize with [threshold_based==True]")

                reg_loss = torch.abs(torch.mean(sorted_y0[(sorted_y0 >= pct_a) & (sorted_y0 <= pct_b)]) - torch.mean(sorted_y1[(sorted_y0 >= pct_a) & (sorted_y0 <= pct_b)]))

            else:
                # Calculate the pct_a and pct_b percentile indices. Regularize on values inbetween.
                index_a_0 = int(pct_a * len_y0)
                index_b_0 = int(pct_b * len_y0)
                index_a_1 = int(pct_a * len_y1)
                index_b_1 = int(pct_b * len_y1)
                reg_loss = torch.abs(torch.mean(sorted_y0[index_a_0:index_b_0]) - torch.mean(sorted_y1[index_a_1:index_b_1]))

        else:
            reg_loss = torch.abs(torch.mean(sorted_y0) - torch.mean(sorted_y1)) #for the whole distribution

        # Return the regularization loss along with three tensors of zeros
        return reg_loss, torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])


# Define a new class called diff_abs_reg that inherits from torch.nn.Module
class diff_abs_reg(torch.nn.Module):
    # Define the constructor for the class
    def __init__(self, mode = "dp", local_reg=True, threshold_based=True):#,lower_percentile=0.5,higher_percentile=1):
        # Call the constructor of the parent class
        super(diff_abs_reg, self).__init__()

        # Set the mode attribute to the value passed in as an argument
        self.mode = mode
        # Set whether regularization happens locally or not
        self.local_reg = local_reg
        # Set whether regularization happens between two cut-off values (True) or two percentiles (False)
        self.threshold_based = threshold_based

    # Define the forward method for the class
    def forward(self, y_pred, s, y_gt, pct_a, pct_b):
        # Select the predicted values corresponding to s == 0 or 1
        y0 = y_pred[s == 0]
        y1 = y_pred[s == 1]

        if self.local_reg:

            # Sort probabilities in ascending order
            sorted_y0, _ = torch.sort(y0)
            sorted_y1, _ = torch.sort(y1)

            len_y0 = len(y0)
            len_y1 = len(y1)

            if self.threshold_based:
                #only select values between 0.7 & 1
                y0 = sorted_y0[(sorted_y0 >= pct_a) & (sorted_y0 <= pct_b)]
                y1 = sorted_y1[(sorted_y1 >= pct_a) & (sorted_y1 <= pct_b)]

                if len(y0) == 0 or len(y1) == 0:
                    raise ValueError(f"At least one group does not have predictions in [{pct_a},  {pct_b}]. Impossible to regularize with [threshold_based==True]")

                reg_loss = diff_abs(y0, y1)

            else:
                #only select values between percentile 0.7 and 1 (per group)
                max_len = max(len_y0,len_y1)

                index_a_0 = int(pct_a*len_y0)
                index_b_0 = int(pct_b*len_y0)
                index_a_1 = int(pct_a*len_y1)
                index_b_1 = int(pct_b*len_y1)

                y0 = sorted_y0[index_a_0:index_b_0]
                y1 = sorted_y1[index_a_1:index_b_1]

                reg_loss = diff_abs(y0, y1)
        else:
            reg_loss = diff_abs(y0, y1) #for the whole distribution

        # Return the regularization loss along with three tensors of zeros
        return reg_loss, torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])



#Takes abs(diff) to regularize. Option to regularize between two percentiles.
def diff_abs(y0, y1):

    len_y0 = len(y0)
    len_y1 = len(y1)

    # Determine the length of the larger tensor
    max_len = max(len_y0, len_y1)

    # Sort probabilities in ascending order
    sorted_y0, _ = torch.sort(y0)
    sorted_y1, _ = torch.sort(y1)

    # Interpolate or pad the smaller tensor to match the length of the larger tensor
    if len_y0 < max_len:
        sorted_y0 = F.interpolate(sorted_y0.unsqueeze(0).unsqueeze(0), size=max_len, mode='linear', align_corners=True).squeeze(0).squeeze(0)
    elif len_y1 < max_len:
        sorted_y1 = F.interpolate(sorted_y1.unsqueeze(0).unsqueeze(0), size=max_len, mode='linear', align_corners=True).squeeze(0).squeeze(0)

    diff = sorted_y0 - sorted_y1

    # Calculate absolute sum of abs differences
    diff_abs = torch.sum(torch.abs(diff))

    return diff_abs


class diff_quadr_reg(torch.nn.Module):
    # Define the constructor for the class
    def __init__(self, mode = "dp", local_reg=True, threshold_based=True):#,lower_percentile=0.5,higher_percentile=1):
        # Call the constructor of the parent class
        super(diff_quadr_reg, self).__init__()

        # Set the mode attribute to the value passed in as an argument
        self.mode = mode
        # Set whether regularization happens between two percentiles (Boolean)
        self.local_reg = local_reg
        # Set whether regularization happens between two cut-off values (True) or two percentiles (False)
        self.threshold_based = threshold_based

    # Define the forward method for the class
    def forward(self, y_pred, s, y_gt, pct_a, pct_b):
        # Select the predicted values corresponding to s == 0 or 1
        y0 = y_pred[s == 0]
        y1 = y_pred[s == 1]

        if self.local_reg:

            # Sort probabilities in ascending order
            sorted_y0, _ = torch.sort(y0)
            sorted_y1, _ = torch.sort(y1)

            len_y0 = len(y0)
            len_y1 = len(y1)

            if self.threshold_based:
                #only select values between 0.7 & 1
                y0 = sorted_y0[(sorted_y0 >= pct_a) & (sorted_y0 <= pct_b)]
                y1 = sorted_y1[(sorted_y1 >= pct_a) & (sorted_y1 <= pct_b)]

                if len(y0) == 0 or len(y1) == 0:
                    raise ValueError(f"At least one group does not have predictions in [{pct_a},  {pct_b}]. Impossible to regularize with [threshold_based==True]")

                reg_loss = diff_quadr(y0, y1)

            else:
                #only select values between percentile 0.7 and 1 (per group)
                max_len = max(len_y0,len_y1)

                index_a_0 = int(pct_a*len_y0)
                index_b_0 = int(pct_b*len_y0)
                index_a_1 = int(pct_a*len_y1)
                index_b_1 = int(pct_b*len_y1)

                y0 = sorted_y0[index_a_0:index_b_0]
                y1 = sorted_y1[index_a_1:index_b_1]

                reg_loss = diff_quadr(y0, y1)
        else:
            reg_loss = diff_quadr(y0, y1) #for the whole distribution

        # Return the regularization loss along with three tensors of zeros
        return reg_loss, torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])


def diff_quadr(y0, y1):
    len_y0 = len(y0)
    len_y1 = len(y1)

    # Determine the length of the larger tensor
    max_len = max(len_y0, len_y1)

    # Sort probabilities in ascending order
    sorted_y0, _ = torch.sort(y0)
    sorted_y1, _ = torch.sort(y1)

    # Interpolate or pad the smaller tensor to match the length of the larger tensor
    if len_y0 < max_len:
        sorted_y0 = F.interpolate(sorted_y0.unsqueeze(0).unsqueeze(0), size=max_len, mode='linear', align_corners=True).squeeze(0).squeeze(0)
    elif len_y1 < max_len:
        sorted_y1 = F.interpolate(sorted_y1.unsqueeze(0).unsqueeze(0), size=max_len, mode='linear', align_corners=True).squeeze(0).squeeze(0)


    # Calculate the differences between the sorted arrays
    diff = sorted_y0 - sorted_y1

    # Calculate the squared differences and sum them up
    diff_quadr = (diff ** 2).sum()

    return diff_quadr



class histogram_reg(torch.nn.Module):
    def __init__(self, mode="dp", local_reg=True, threshold_based=True, bin_width=0.05):
        super(histogram_reg, self).__init__()
        self.mode = mode
        self.local_reg = local_reg
        # Set whether regularization happens between two cut-off values (True) or two percentiles (False)
        self.threshold_based = threshold_based
        self.bin_width = bin_width

    ############################################
    # Differentiable Histogram Counting Method
    #############################################


    def forward(self, y_pred, s, y_gt, pct_a, pct_b):

        bin_width = self.bin_width

        y0 = y_pred[s == 0]
        y1 = y_pred[s == 1]

        len_y0 = len(y0)
        len_y1 = len(y1)

        y0 = y0.reshape(1, len_y0)
#        y0.requires_grad = True
        y1 = y1.reshape(1, len_y1)
#        y1.requires_grad = True

        # Compute histogram (differentiable implementation)
        bins = int(1/bin_width)
        hist_differentiable0 = differentiable_histogram(y0, bins=bins, min=0.0, max=1.0)[0, 0, :]
        hist_differentiable1 = differentiable_histogram(y1, bins=bins, min=0.0, max=1.0)[0, 0, :]

        # Normalize
        #hist_differentiable0 = torch.div(hist_differentiable0, len(y0))
        hist_differentiable0 = hist_differentiable0 / torch.sum(hist_differentiable0)
        #hist_differentiable1 = torch.div(hist_differentiable1, len(y1))
        hist_differentiable1 = hist_differentiable1 / torch.sum(hist_differentiable1)


        if self.local_reg:
            if self.threshold_based:
                pct_a_bin = int(bins * pct_a)
                pct_b_bin = int(bins * pct_b)

                #set values outside pct_a and pct_b to zero
                hist_differentiable0[:pct_a_bin] = 0
                hist_differentiable0[pct_b_bin:] = 0

                hist_differentiable1[:pct_a_bin] = 0
                hist_differentiable1[pct_b_bin:] = 0

                reg_loss = torch.sum(torch.abs(hist_differentiable0 - hist_differentiable1))
                #reg_loss = torch.sum((hist_differentiable0 - hist_differentiable1)**2)

            else:
                # only select values between percentile 0.7 and 1 (per group)
                cumulative_hist0 = torch.cumsum(hist_differentiable0, dim=0)
                cumulative_hist1 = torch.cumsum(hist_differentiable1, dim=0)

                decision_bins_0 = torch.where((cumulative_hist0 >= pct_a) & (cumulative_hist0 <= pct_b))[0]
                decision_bins_1 = torch.where((cumulative_hist1 >= pct_a) & (cumulative_hist1 <= pct_b))[0]

                mask0 = torch.zeros_like(hist_differentiable0)
                mask0[decision_bins_0] = 1

                mask1 = torch.zeros_like(hist_differentiable1)
                mask1[decision_bins_1] = 1

                # Use the mask to set values not in 'decision_bins_0/1' to 0
                hist_differentiable0 *= mask0
                hist_differentiable1 *= mask1

                reg_loss = torch.sum(torch.abs(hist_differentiable0 - hist_differentiable1))


                #raise ValueError("percentile-based approach not implemented for histogram-based regularization. 'threshold_based' should be set to True")

        else:
            reg_loss = torch.sum(torch.abs(hist_differentiable0 - hist_differentiable1))

        #print(reg_loss)

        return reg_loss, torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])


class histogram_sq_reg(torch.nn.Module):
    def __init__(self, mode="dp", local_reg=True, threshold_based=True, bin_width=0.05):
        super(histogram_sq_reg, self).__init__()
        self.mode = mode
        self.local_reg = local_reg
        # Set whether regularization happens between two cut-off values (True) or two percentiles (False)
        self.threshold_based = threshold_based
        self.bin_width = bin_width

    ############################################
    # Differentiable Histogram Counting Method
    #############################################

    def forward(self, y_pred, s, y_gt, pct_a, pct_b):

        bin_width = self.bin_width

        y0 = y_pred[s == 0]
        y1 = y_pred[s == 1]

        len_y0 = len(y0)
        len_y1 = len(y1)

        y0 = y0.reshape(1, len_y0)
        #        y0.requires_grad = True
        y1 = y1.reshape(1, len_y1)
        #        y1.requires_grad = True

        # Compute histogram (differentiable implementation)
        bins = int(1 / bin_width)
        hist_differentiable0 = differentiable_histogram(y0, bins=bins, min=0.0, max=1.0)[0, 0, :]
        hist_differentiable1 = differentiable_histogram(y1, bins=bins, min=0.0, max=1.0)[0, 0, :]

        # Normalize
        # hist_differentiable0 = torch.div(hist_differentiable0, len(y0))
        hist_differentiable0 = hist_differentiable0 / torch.sum(hist_differentiable0)
        # hist_differentiable1 = torch.div(hist_differentiable1, len(y1))
        hist_differentiable1 = hist_differentiable1 / torch.sum(hist_differentiable1)

        if self.local_reg:
            if self.threshold_based:
                pct_a_bin = int(bins * pct_a)
                pct_b_bin = int(bins * pct_b)

                # set values outside pct_a and pct_b to zero
                hist_differentiable0[:pct_a_bin] = 0
                hist_differentiable0[pct_b_bin:] = 0

                hist_differentiable1[:pct_a_bin] = 0
                hist_differentiable1[pct_b_bin:] = 0

                reg_loss = torch.sum(torch.square(hist_differentiable0 - hist_differentiable1))
                # reg_loss = torch.sum((hist_differentiable0 - hist_differentiable1)**2)

            else:
                # only select values between percentile 0.7 and 1 (per group)
                cumulative_hist0 = torch.cumsum(hist_differentiable0, dim=0)
                cumulative_hist1 = torch.cumsum(hist_differentiable1, dim=0)

                decision_bins_0 = torch.where((cumulative_hist0 >= pct_a) & (cumulative_hist0 <= pct_b))[0]
                decision_bins_1 = torch.where((cumulative_hist1 >= pct_a) & (cumulative_hist1 <= pct_b))[0]

                mask0 = torch.zeros_like(hist_differentiable0)
                mask0[decision_bins_0] = 1

                mask1 = torch.zeros_like(hist_differentiable1)
                mask1[decision_bins_1] = 1

                # Use the mask to set values not in 'decision_bins_0/1' to 0
                hist_differentiable0 *= mask0
                hist_differentiable1 *= mask1

                reg_loss = torch.sum(torch.square(hist_differentiable0 - hist_differentiable1))

                # raise ValueError("percentile-based approach not implemented for histogram-based regularization. 'threshold_based' should be set to True")

        else:
            reg_loss = torch.sum(torch.square(hist_differentiable0 - hist_differentiable1))

        return reg_loss, torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])

def differentiable_histogram(x, min=0.0, max=1.0, bin_width=0.05):
  nodes = int((max - min) / bin_width + 1)
  hist_torch = torch.zeros(nodes).to(x.device)

  BIN_Table = torch.arange(start=-1, end=nodes+1, step=1) * bin_width
  #print(BIN_Table)
  #print(BIN_Table.shape)

  for node in range(1, nodes+1, 1): # 1,2,...,nodes
    t_r = BIN_Table[node].item()
    t_r_sub_1 = BIN_Table[node - 1].item()
    t_r_plus_1 = BIN_Table[node + 1].item()

    mask_sub = ((t_r > x) & (x >= t_r_sub_1)).float()
    mask_plus = ((t_r_plus_1 > x) & (x >= t_r)).float()

    #print(torch.count_nonzero(mask_sub))
    #print(torch.count_nonzero(mask_plus))

    # soft assignment
    hist_torch[node-1] += torch.sum(((x - t_r_sub_1)/bin_width * mask_sub), dim=-1)
    hist_torch[node-1] += torch.sum(((t_r_plus_1 - x)/bin_width * mask_plus), dim=-1)

  hist_torch_relative_freq = hist_torch / torch.sum(hist_torch)

  return hist_torch_relative_freq