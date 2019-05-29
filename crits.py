from utils.misc import *

# loss and metrics
l1, l2 = nn.L1Loss(), nn.MSELoss()

class MAPD(nn.Module):
    def __init__(self, stats=None, n_cont=2):
        self.stats,self.n_cont = stats,n_cont
        if n_cont is not None:
            self.stats = [o[:n_cont] for o in self.stats]
        super().__init__()

    def forward(self, input, target):
        if self.n_cont is not None:
            input, target = (o[:,:self.n_cont] for o in (input,target))
        if self.stats is not None:
            input, target = (denormalize(o, *self.stats) for o in (input, target))

        return ((input - target).abs() / (target + eps)).mean()

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

class DualLoss(nn.Module):
    def __init__(self, n_cont:int, n_cat: List[int], lmbd=1):
        super().__init__()
        self.n_cont,self.n_cat,self.lmbd = n_cont,n_cat,lmbd
        self.cat_idx = np.cumsum(n_cat) # where to cut cat variables in prediction
        # self.loss_cont = nn.MSELoss()
        self.loss_cont = MSELossFlat()
        self.loss_cat = nn.CrossEntropyLoss()

    def forward(self, input, target):
        p_cont, p_cat = input[:,:self.n_cont], input[:,self.n_cont:]
        p_cat = torch.sigmoid(p_cat)
        y_cont, y_cat = target[:,:self.n_cont], target[:,self.n_cont:].long()
        # seperate p_cat
        p_cat_tasks = []
        cat_idx_zip = [0] + self.cat_idx.tolist()
        for i,j in zip(cat_idx_zip[:-1], cat_idx_zip[1:]):
            p_cat_tasks.append(p_cat[:,i:j])

        # cont loss
        cont_loss = self.loss_cont(p_cont, y_cont) 
        # cat loss
        cat_loss = 0
        for p, y in zip(p_cat_tasks, y_cat.transpose(0,1)):
            cat_loss += self.loss_cat(p, y) 
        return cont_loss + self.lmbd * cat_loss

