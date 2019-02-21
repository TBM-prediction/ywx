from crits import *
from utils.imports import *

class MultiInputSequentialRNN(nn.Sequential):
    "A sequential module that passes the reset call to its children."
    def forward(self, *x):
        return super().forward(torch.cat([x[0][:,None].float(), x[1]], 1))

    def reset(self):
        for c in self.children():
            if hasattr(c, 'reset'): c.reset()

class ContModel1(RNNCore):
    def __init__(self, n_cat:int, n_cont:int, n_hid:int, n_layers:int, sl=30, 
                 bidir:bool=False, hidden_p:float=0.2, input_p:float=0.6, 
                 embed_p:float=0.1, weight_p:float=0.5, qrnn:bool=False):
        vocab_sz,pad_token=1,0 # continuous variables only for this model
        self.sl, self.n_cat, self.n_cont = sl, n_cat, n_cont
        
        super().__init__(vocab_sz=vocab_sz, emb_sz=n_cont, n_hid=n_hid, n_layers=n_layers, pad_token=pad_token, bidir=bidir,
                 hidden_p=hidden_p, input_p=input_p, embed_p=embed_p, weight_p=weight_p, qrnn=qrnn)
        
    def forward(self, x)->Tuple[Tensor,Tensor]:
        x_cat, x_cont = x[:,self.n_cat], x[:,self.n_cat:]
        bs,_ = x_cont.size()
        input = x_cont.view(bs, self.sl, self.n_cont)
        self.reset()

        if bs!=self.bs:
            self.bs=bs
            self.reset()
        raw_output = self.input_dp(input)
        new_hidden,raw_outputs,outputs = [],[],[]
        for l, (rnn,hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            raw_output, new_h = rnn(raw_output, self.hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.n_layers - 1: raw_output = hid_dp(raw_output)
            outputs.append(raw_output)
        self.hidden = to_detach(new_hidden, cpu=False)
        return raw_outputs, outputs

def get_cont_model(context, databunch, hyper_params):
    rnn_enc = ContModel1(1, context.n_cont, hyper_params['n_hidden'], hyper_params['n_layers'], sl=context.sl,
                         hidden_p=hyper_params['hidden_p'], input_p=hyper_params['input_p'], embed_p=0, weight_p=hyper_params['weight_p'])
    model = MultiInputSequentialRNN(rnn_enc, PoolingLinearClassifier(hyper_params['layers'], hyper_params['drops'])).cuda()

    learner = Learner(databunch, model, loss_func=hyper_params['loss_func'], metrics=context.metrics, opt_func=optim.SGD)

    # learner.callback_fns += [ShowGraph, partial(SaveModelCallback, name='rnn0')]
    learner.callback_fns += [ShowGraph,]
    learner.callbacks.append(TerminateOnNaNCallback())
    learner.callbacks.append(RNNTrainer(learner, context.sl, alpha=hyper_params['alpha'], beta=hyper_params['beta']))

    learner.callback_fns.append(partial(GradientClipping, clip=hyper_params['clip']))
    learner.split(rnn_classifier_split)
    return learner

