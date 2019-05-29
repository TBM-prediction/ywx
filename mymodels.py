from crits import *
from utils.imports import *

# class MultiInputSequentialRNN(nn.Sequential):
    # "A sequential module that passes the reset call to its children."
    # def forward(self, *x):
        # return super().forward(torch.cat([x[0][:,None].float(), x[1]], 1))

    # def reset(self):
        # for c in self.children():
            # if hasattr(c, 'reset'): c.reset()

# class ContModel1(RNNCore):
    # def __init__(self, n_cat:int, n_cont:int, n_hid:int, n_layers:int, sl=30,
                 # bidir:bool=False, hidden_p:float=0.2, input_p:float=0.6,
                 # embed_p:float=0.1, weight_p:float=0.5, qrnn:bool=False):
        # vocab_sz,pad_token=1,0 # continuous variables only for this model
        # self.sl, self.n_cat, self.n_cont = sl, n_cat, n_cont

        # super().__init__(vocab_sz=vocab_sz, emb_sz=n_cont, n_hid=n_hid, n_layers=n_layers, pad_token=pad_token, bidir=bidir,
                 # hidden_p=hidden_p, input_p=input_p, embed_p=embed_p, weight_p=weight_p, qrnn=qrnn)

    # def forward(self, x)->Tuple[Tensor,Tensor]:
        # x_cat, x_cont = x[:,self.n_cat], x[:,self.n_cat:]
        # bs,_ = x_cont.size()
        # input = x_cont.view(bs, self.sl, self.n_cont)
        # self.reset()

        # if bs!=self.bs:
            # self.bs=bs
            # self.reset()
        # raw_output = self.input_dp(input)
        # new_hidden,raw_outputs,outputs = [],[],[]
        # for l, (rnn,hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            # raw_output, new_h = rnn(raw_output, self.hidden[l])
            # new_hidden.append(new_h)
            # raw_outputs.append(raw_output)
            # if l != self.n_layers - 1: raw_output = hid_dp(raw_output)
            # outputs.append(raw_output)
        # self.hidden = to_detach(new_hidden, cpu=False)
        # return raw_outputs, outputs


# def get_cont_model(context, databunch, hyper_params):
    # rnn_enc = ContModel1(1, context.n_cont, hyper_params['n_hidden'], hyper_params['n_layers'], sl=context.sl,
                         # hidden_p=hyper_params['hidden_p'], input_p=hyper_params['input_p'], embed_p=0, weight_p=hyper_params['weight_p'])
    # model = MultiInputSequentialRNN(rnn_enc, PoolingLinearClassifier(hyper_params['layers'], hyper_params['drops'])).cuda()

    # learner = Learner(databunch, model, loss_func=hyper_params['loss_func'], metrics=context.metrics, opt_func=optim.SGD)

    # # learner.callback_fns += [ShowGraph, partial(SaveModelCallback, name='rnn0')]
    # learner.callback_fns += [ShowGraph,]
    # learner.callbacks.append(TerminateOnNaNCallback())
    # learner.callbacks.append(RNNTrainer(learner, context.sl, alpha=hyper_params['alpha'], beta=hyper_params['beta']))

    # learner.callback_fns.append(partial(GradientClipping, clip=hyper_params['clip']))
    # learner.split(rnn_classifier_split)
    # return learner

class PoolingLinearClassifier(nn.Module):
    "Create a linear classifier with pooling."

    def __init__(self, layers:Collection[int], drops:Collection[float],
            use_extra_x:bool=False):
        super().__init__()
        self.use_extra_x = use_extra_x
        mod_layers = []
        activs = [nn.ReLU(inplace=True)] * (len(layers) - 2) + [None]
        for n_in,n_out,p,actn in zip(layers[:-1],layers[1:], drops, activs):
            mod_layers += bn_drop_lin(n_in, n_out, p=p, actn=actn)
        self.layers = nn.Sequential(*mod_layers)

    def pool(self, x:Tensor, bs:int, is_max:bool):
        "Pool the tensor along the seq_len dimension."
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.transpose(1,2), (1,)).view(bs,-1)

    def forward(self, input:Tuple[Tensor,Tensor])->Tuple[Tensor,Tensor,Tensor]:
        raw_outputs, outputs, extra_x = input
        output = outputs[-1]
        bs,sl,_ = output.size()
        avgpool = self.pool(output, bs, False)
        mxpool = self.pool(output, bs, True)
        x = torch.cat([output[:,-1], mxpool, avgpool], 1)
        if self.use_extra_x:
            x = torch.cat([x, extra_x], 1)
        x = self.layers(x)
        return x, raw_outputs, outputs

class Task2Model(AWD_LSTM):
    def __init__(self, n_cont, emb_sz:int, n_hid:int, n_layers:int, layers:Collection[int], drops:Collection[float], hidden_p:float=0.2,
                 input_p:float=0.6, embed_p:float=0.1, weight_p:float=0.5, bidir:bool=False, 
                 sl=30, use_extra_x=False, use_linear_decoder=False):
        # note: check pad_token when generating cat variables
        qrnn = False # continuous variables only for this model
        vocab_sz,pad_token=1,0 # temp
        super().__init__(vocab_sz, n_cont, n_hid, n_layers, pad_token, hidden_p, input_p, embed_p, weight_p, qrnn, bidir)
        self.sl,self.n_cont,self.emb_sz = sl,n_cont,emb_sz
        if use_linear_decoder:
            self.decoder = PoolingLinearClassifier(layers, drops, use_extra_x=use_extra_x)
        else: # TODO
            self.decoder = PoolingLinearClassifier(layers, drops, use_extra_x=use_extra_x)
        
        # re-initialize embedding layer
        vocab_sz,pad_token=1,0
        self.encoder = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_token)
        self.encoder_dp = EmbeddingDropout(self.encoder, embed_p)
        
    def forward(self, x_cat:Tensor, x_cont:Tensor):
        bs,*_ = x_cat.size()
        if bs!=self.bs:
            self.bs=bs
            self.reset()
        x_cont, x_extra = x_cont[:,:self.n_cont*self.sl], x_cont[:,self.n_cont*self.sl:]
        x_cont = x_cont.view(bs, self.sl, self.n_cont)
        x_cat = self.encoder_dp(x_cat)

        raw_output = self.input_dp(x_cont)
        new_hidden,raw_outputs,outputs = [],[],[]
        for l, (rnn,hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            raw_output, new_h = rnn(raw_output, self.hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.n_layers - 1: raw_output = hid_dp(raw_output)
            outputs.append(raw_output)
        self.hidden = to_detach(new_hidden, cpu=False)
        
        # fc
        x, raw_outputs, outputs = self.decoder((raw_outputs, outputs, x_extra))
        return x, raw_outputs, outputs
    
    def _one_hidden(self, l:int)->Tensor:
        "Return one hidden state."
        nh = (self.n_hid if l != self.n_layers - 1 else self.n_cont) // self.n_dir
        return one_param(self).new(1, self.bs, nh).zero_()

def get_new_model(context, databunch, hyper_params):
    emb_sz = 3 # use rule of thumb later
    model = Task2Model(context.n_cont,emb_sz, hyper_params['n_hidden'], hyper_params['n_layers'], hyper_params['layers'], hyper_params['drops'],
		      hidden_p=hyper_params['hidden_p'],
                      input_p=hyper_params['input_p'],
                      weight_p=hyper_params['weight_p'],
                      use_extra_x=hyper_params['use_extra_x'])
    model = model.cuda()

    learner = Learner(databunch, model, loss_func=hyper_params['loss_func'], metrics=context.metrics, opt_func=optim.SGD)

    # learner.callback_fns += [ShowGraph, partial(SaveModelCallback, name='rnn0')]
    learner.callback_fns += [ShowGraph,]
    learner.callbacks.append(TerminateOnNaNCallback())
    learner.callbacks.append(RNNTrainer(learner, alpha=hyper_params['alpha'], beta=hyper_params['beta']))

    learner.callback_fns.append(partial(GradientClipping, clip=hyper_params['clip']))
    # learner.split(rnn_classifier_split)
    return learner


