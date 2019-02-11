from utils.imports import *

# loss and metrics
eps= 1e-7
l1, l2 = nn.MSELoss(), nn.L1Loss()

def mapd(input, target):
    #return torch.abs((input-target) / (target+eps)).sum() / target.view(-1).size()[0]
    #set_trace()
    return (1 - (input/(eps+target)).mean()).abs()

def our_metrics(input, target):
    # score range: (-NaN, 2]
    return l1(input, target)
    # return 2 - (torch.abs(input-target) / (target+eps)).sum() / target.view(-1).size()[0]

def our_metrics_np(input, target):
    # score range: (-NaN, 2]
    return 2 - (np.abs(input-target) / (target+eps)).sum() / target.size

def rnn_metrics(input, target):
    #return -our_metrics(input[0], target)
    return mapd(input[0], target)

def weighted_our_loss(weight):
    def loss(input, target): return rnn_metrics([weight * input[0]], weight * target)
    print(tile.mean())
    return loss

class WeightedRNNMSE:
    def __init__(self, weight):
        self.weight = weight
        
    def __call__(self, input, target):
        return l2(self.weight * input, self.weight * target)

# def our_log_loss(input, target):
    # return l2(input, (target+eps).log())

def mean_std_np(x):
    return x.mean(), x.std()


