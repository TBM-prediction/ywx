from utils.misc import *

# loss and metrics
l1, l2 = nn.L1Loss(), nn.MSELoss()

class MAPD(nn.Module):
    def __init__(self, stats=None):
        self.stats = stats
        super().__init__()

    def forward(self, input, target):
        if self.stats is not None:
            input, target = (denormalize(o, *self.stats) for o in (input, target))

        if isinstance(input, torch.Tensor):
            return (1 - (input/(eps+target)).mean()).abs()
        else:
            return np.abs(1 - (input/(eps+target)).mean())

    # to inform fastai library the name of the metrics
    @property
    def __name__(self):
        return self.__class__.__name__

def our_metrics(input, target):
    # score range: (-NaN, 2]
    return l1(input, target)

def our_metrics_np(input, target):
    # score range: (-NaN, 2]
    return 2 - (np.abs(input-target) / (target+eps)).sum() / target.size

def rnn_metrics(input, target):
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

def mean_std_np(x):
    return x.mean(), x.std()


