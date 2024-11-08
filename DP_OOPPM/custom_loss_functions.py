
"""Contents:

note: class definitions end in '_reg'

gap_reg: demographic parity loss - WORKS

Wasserstein_reg (simple implementation as 'proxy' for ABPC regularization) - WORKS

ABCC_reg (#TODO: DOES NOT WORK YET)



"""


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance


"""
# Define a new class called ABCC_reg that inherits from torch.nn.Module
class ABCC_reg(torch.nn.Module):
    # Define the constructor for the class
    def __init__(self, mode = "dp"):
        # Call the constructor of the parent class
        super(ABCC_reg, self).__init__()
        # Set the mode attribute to the value passed in as an argument
        self.mode = mode

 #       self.fair_loss = ABCC_reg_loss  # Register ABCC_reg_loss as a submodule


    # Define the forward method for the class
    def forward(self, y_pred, s, y_gt):
        # Select the predicted values corresponding to s == 0 and 1
        y0 = y_pred[s == 0]
        y1 = y_pred[s == 1]

#        reg_loss = wasserstein_distance(y0,y1)
        reg_loss = ABCC_reg_loss( y0, y1, s, sample_n = 10000)

#       print(reg_loss)

        # Return the regularization loss along with three tensors of zeros
        return reg_loss, torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])


class ECDF_torch:
    def __init__(self, x, side='right'):
        self.x = torch.tensor(x, requires_grad=True)
        x_sorted, _ = torch.sort(self.x)
        nobs = len(self.x)
        self.y = torch.linspace(1. / nobs, 1, nobs, requires_grad=True)
        self.side = side

#        x = torch.tensor(x, requires_grad = True)
#        x_sorted, _ = torch.sort(x)
#        nobs = len(x)
#        y = torch.linspace(1./nobs, 1, nobs, requires_grad = True)
#        self.x = x_sorted
#        self.y = y
#        self.side = side

    def __call__(self, values):
        values = torch.tensor(values, requires_grad = True)
        if self.side == 'right':
            cdf = torch.searchsorted(self.x, values, right=True) / len(self.x)
        elif self.side == 'left':
            cdf = torch.searchsorted(self.x, values, right=False) / len(self.x)
        else:
            raise ValueError("Invalid value for 'side'. It must be either 'left' or 'right'.")
        print(cdf)
        return cdf

    def plot(self):
        plt.step(self.x, self.y, where='post')
        plt.xlabel('Values')
        plt.ylabel('ECDF')
        plt.title('Empirical CDF')
        plt.show()


# y_pred: predicted values
# s: binary values indicating the group membership of each sample
# sample_n: number of samples to generate for the integration (default is 10000)
def ABCC_reg_loss( y0, y1, s, sample_n = 10000 ):
    # Todo: error: een object van class ECDF_torch heeft geen grad - op een manier oplossen?
    # Flatten the input arrays
#    y_pred = y_pred.ravel()
#    s = s.ravel()

    # Extract the predicted values for each group
#    y_pre_1 = y_pred[s == 1]
#    y_pre_0 = y_pred[s == 0]

    # Compute the empirical cumulative distribution function (ECDF) for each group
#    print(y0.requires_grad)


    ecdf0 = ECDF_torch(y0)
 #   ecdf0.x.requires_grad_(True)
    ecdf0.y0.requires_grad_(True)
#    print(ecdf0.requires_grad)

 #   print(ecdf0)
    ecdf1 = ECDF_torch(y1)

    # Plot ECDF for ecdf0
#   ecdf0.plot()

    # Plot ECDF for ecdf1
#    ecdf1.plot()

    # Generate a set of x values for the integration
    x = torch.linspace(0, 1, 10000)

    # Evaluate the ECDFs at the x values
    ecdf0_x = ecdf0(x)
    ecdf1_x = ecdf1(x)

 #   print(ecdf0_x)

    # Compute the area between the two ECDFs using the trapezoidal rule
    with torch.enable_grad():
        abcc_loss = torch.trapz(torch.abs(ecdf0_x - ecdf1_x), x)

    # Return the computed ABCC value
    return abcc_loss
"""


# Define a new class called gap_reg that inherits from torch.nn.Module
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



    """
    #TODO: this definition contains two versions: A. calculate top k% on whole population. and B. calculate top k% per subgroup
    #TODO: B. this version takes k% per protected group.
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


    # Calculate the pct_a and pct_b percentile indices. Regularize on values inbetween.
    index_a = int(pct_a * max_len)
    index_b = int(pct_b * max_len)

    # Calculate the differences between the sorted arrays
    diff = sorted_y0[index_a:index_b] - sorted_y1[index_a:index_b] #if no percentiles are given, difference is calculated in [0,1]

    # Calculate the pseudo-Wasserstein distance
    diff_abs = torch.sum(torch.abs(diff))# * torch.arange(1, len(diff) + 1))

    return diff_abs
    """

    """
    #TODO: A. this version calculates top k% on whole population
    #concatenate two tensors
    y_pred = torch.cat((y1, y0), dim=0)

    # Calculate the percentile indices
    k_a = int(pct_a * (y_pred.numel() -1))
    k_b = int(pct_b * (y_pred.numel() -1))

    k_a = max(k_a,1)
    # Check if percentile indices are out of range
    if k_a < 0 or k_b >= y_pred.numel():
        raise ValueError("Percentile indices are out of range.")

    # Calculate the percentile values
    percentile_a = torch.kthvalue(y_pred, k_a, keepdim=False).values
    percentile_b = torch.kthvalue(y_pred, k_b, keepdim=False).values

#    percentile_a = torch.kthvalue(y_pred, int(pct_a * y_pred.numel()), keepdim=False).values
#    percentile_b = torch.kthvalue(y_pred, int(pct_b * y_pred.numel()), keepdim=False).values

    y0_local = y0[percentile_a < y0]
    y0_local = y0_local[y0_local < percentile_b] #todo: cleaner in one operation
    y1_local = y1[percentile_a < y1]
    y1_local = y1_local[y1_local < percentile_b]

    len_y0_local = len(y0_local)     #todo: add check that len of either arrays cannot be 0? Otherwise impossible to extrapolate
    len_y1_local = len(y1_local)

    # Determine the length of the larger tensor
    max_len = max(len_y0_local, len_y1_local)

    # Sort probabilities in ascending order
    sorted_y0_local, _ = torch.sort(y0_local)
    sorted_y1_local, _ = torch.sort(y1_local)

    # Interpolate or pad the smaller tensor to match the length of the larger tensor
    if len_y0_local < max_len:
        sorted_y0_local = F.interpolate(sorted_y0_local.unsqueeze(0).unsqueeze(0), size=max_len, mode='linear', align_corners=True).squeeze(0).squeeze(0)
    elif len_y1_local < max_len:
        sorted_y1_local = F.interpolate(sorted_y1_local.unsqueeze(0).unsqueeze(0), size=max_len, mode='linear', align_corners=True).squeeze(0).squeeze(0)

    # Calculate the differences between the sorted arrays
    diff = sorted_y0_local - sorted_y1_local #if no percentiles are given, difference is calculated in [0,1]

    # Calculate the pseudo-Wasserstein distance
    wasserstein = torch.sum(torch.abs(diff))# * torch.arange(1, len(diff) + 1))
    """




class wasserstein_reg(torch.nn.Module):
    """
    As implemented by Shalit et al., translated to PyTorch: https://github.com/clinicalml/cfrnet/blob/master/cfr/util.py
    """
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

        y0 = y_pred[s == 0]
        y1 = y_pred[s == 1]

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
#        a = torch.cat((p * torch.ones((s.sum().item(), 1), device=y_pred.device) / nt, (1 - p) * torch.ones((1, 1), device=y_pred.device)))
#        a = torch.cat((p * torch.ones((s.sum().item(), 1), device=y_pred.device) / nt,(1 - p) * torch.ones((s.sum().item(), 1), device=y_pred.device)))

        a_indices = torch.where(s > 0)[0]
        #a = torch.cat([(p * torch.ones(a_indices.shape[0]) / nt).unsqueeze(1), (1 - p) * torch.ones((1, 1))], dim=0)
        a = torch.cat([(p * torch.ones(len(y1)) / nt).unsqueeze(1), (1 - p) * torch.ones((1, 1))], dim=0)

        #b = torch.cat(((1 - p) * torch.ones((nc, 1), device=y_pred.device) / nc, p * torch.ones((1, 1), device=y_pred.device)))
        b_indices = torch.where(s < 1)[0]
        #b = torch.cat([((1 - p) * torch.ones(b_indices.shape[0]) / nc).unsqueeze(1), p * torch.ones((1, 1))], dim = 0)
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

        """
        #Check for similarity with scipy implementation:
        dist_dirt = y0.detach().numpy()
        dist_holes = y1.detach().numpy()
        wd = wasserstein_distance(dist_dirt, dist_holes)
        print(wd/float(reg_loss))
        #print('Wass scipy: '+str(wd))
        #print('Wass own: '+str(float(reg_loss)))
        """

        return reg_loss, torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])


class KL_divergence_reg(torch.nn.Module):
    def __init__(self, mode="dp", local_reg=True, threshold_based=True):
        super(KL_divergence_reg, self).__init__()
        self.mode = mode
        self.local_reg = local_reg
        # Set whether regularization happens between two cut-off values (True) or two percentiles (False)
        self.threshold_based = threshold_based

    def forward(self, y_pred, s, y_gt, pct_a, pct_b):
        y0 = y_pred[s == 0]
        y1 = y_pred[s == 1]

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


        """
        # Select the predicted values corresponding to s == 0 or 1
        y0 = y_pred[s == 0]
        y1 = y_pred[s == 1]

        if self.local_reg:
            reg_loss = diff_quadr(y0, y1,pct_a,pct_b)
        else:
            reg_loss = diff_quadr(y0, y1,pct_a=0,pct_b=1) #for the whole distribution

        # Return the regularization loss along with three tensors of zeros
        return reg_loss, torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])
        """

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

    """
    # Calculate the pct_a and pct_b percentile indices. Regularize on values inbetween.
    index_a = int(pct_a * max_len)
    index_b = int(pct_b * max_len)
    """

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

        """
        plt.subplot(122)
        plt.plot(hist_differentiable0.detach().numpy())
        plt.show()

        plt.subplot(122)
        plt.plot(hist_differentiable1.detach().numpy())
        plt.show()
        """

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

        """
        if self.local_reg:

            if self.threshold_based:
                # only select values between 0.7 & 1
                decision_area = torch.linspace(pct_a, pct_b, 1000)

            #    if len(y0) == 0 or len(y1) == 0:
            #        raise ValueError(f"At least one group does not have predictions in [{pct_a},  {pct_b}]. "
            #                         f"Impossible to regularize with [threshold_based==True]")

            #Todo:
          
            else: 
                # only select values between percentile 0.7 and 1 (per group)
                index_a_0 = int(pct_a*len_y0)
                index_b_0 = int(pct_b*len_y0)
                index_a_1 = int(pct_a*len_y1)
                index_b_1 = int(pct_b*len_y1)

                y0 = y0[index_a_0:index_b_0]
                y1 = y1[index_a_1:index_b_1]
    
        else:
            decision_area = torch.linspace(0, 1, 1000)

        len_y0 = len(y0)
        len_y1 = len(y1)

    
        print(hist0)

        x = range(20)
        plt.bar(x, hist0, align='center', color=['forestgreen'])
        plt.bar(range(0.5,20.5,1), hist1, align='center', color=['red'])
        plt.xlabel('Bins')
        plt.ylabel('Frequency')
        plt.show()
        
        """

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

        """
        plt.subplot(122)
        plt.plot(hist_differentiable0.detach().numpy())
        plt.show()

        plt.subplot(122)
        plt.plot(hist_differentiable1.detach().numpy())
        plt.show()
        """

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

        # print(reg_loss)

        """
        if self.local_reg:

            if self.threshold_based:
                # only select values between 0.7 & 1
                decision_area = torch.linspace(pct_a, pct_b, 1000)

            #    if len(y0) == 0 or len(y1) == 0:
            #        raise ValueError(f"At least one group does not have predictions in [{pct_a},  {pct_b}]. "
            #                         f"Impossible to regularize with [threshold_based==True]")

            #Todo:

            else: 
                # only select values between percentile 0.7 and 1 (per group)
                index_a_0 = int(pct_a*len_y0)
                index_b_0 = int(pct_b*len_y0)
                index_a_1 = int(pct_a*len_y1)
                index_b_1 = int(pct_b*len_y1)

                y0 = y0[index_a_0:index_b_0]
                y1 = y1[index_a_1:index_b_1]

        else:
            decision_area = torch.linspace(0, 1, 1000)

        len_y0 = len(y0)
        len_y1 = len(y1)


        print(hist0)

        x = range(20)
        plt.bar(x, hist0, align='center', color=['forestgreen'])
        plt.bar(range(0.5,20.5,1), hist1, align='center', color=['red'])
        plt.xlabel('Bins')
        plt.ylabel('Frequency')
        plt.show()

        """

        return reg_loss, torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])


# def differentiable_histogram(x, bins=20, min=0.0, max=1.0):

#     if len(x.shape) == 4:
#         n_samples, n_chns, _, _ = x.shape
#     elif len(x.shape) == 2:
#         n_samples, n_chns = 1, 1
#     else:
#         raise AssertionError('The dimension of input tensor should be 2 or 4.')

#     hist_torch = torch.zeros(n_samples, n_chns, bins).to(x.device)
#     delta = (max - min) / bins

#     BIN_Table = torch.arange(start=0, end=bins, step=1) * delta

#     for dim in range(1, bins - 1, 1):
#         h_r = BIN_Table[dim].item()  # h_r
#         h_r_sub_1 = BIN_Table[dim - 1].item()  # h_(r-1)
#         h_r_plus_1 = BIN_Table[dim + 1].item()  # h_(r+1)

#         mask_sub = ((h_r > x) & (x >= h_r_sub_1)).float()
#         mask_plus = ((h_r_plus_1 > x) & (x >= h_r)).float()

#         hist_torch[:, :, dim] += torch.sum(((x - h_r_sub_1) * mask_sub).view(n_samples, n_chns, -1), dim=-1)
#         hist_torch[:, :, dim] += torch.sum(((h_r_plus_1 - x) * mask_plus).view(n_samples, n_chns, -1), dim=-1)

#     return hist_torch / delta

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
