from .imports import *

eps = float(1e-7)
def zero_boundary(df):
    zeros = (df == 0).astype('int')
    boundary = zeros.diff()

    # handle boundary condition (no pun intended)
    if zeros.iloc[0] == 1: boundary.iloc[0] = 1
    if zeros.iloc[-1] == 1: boundary.iloc[-1] = -1

    begins = np.argwhere(np.array((boundary == 1))).squeeze()
    ends = np.argwhere(np.array((boundary == -1))).squeeze()

    return begins, ends

def get_interesting_columns(df, merge=True, print_=False):
    columns_list = df.columns.tolist()
    params = set()
    for target_param in ('推进速度', '刀盘转速', '推进力', '扭矩', '功率'):
        for param in columns_list:
            if target_param in param:
                params.add(param)
    df = df.loc[:,params]

    if merge:
        import re
        for reg in ('主驱动\d+#电机扭矩', '主驱动\d+#电机输出功率'):
            merge_params = {o for o in params if re.match(reg, o)}
            rename = reg.replace('\d+#', '')
            params -= merge_params
            params.add(rename)
            df = pd.concat([df, df.loc[:,merge_params].mean(axis=1).rename(rename)], axis=1, copy=False)
            df = df.drop(columns=merge_params)

    if print_:
        print(params)
    
    return df

def rolling_cumsum(df):
    df = df.loc[:,'刀盘扭矩'].rolling(10, center=True).mean()
    diff = df.diff()
    diff.loc[diff<0] = 0
    return diff.cumsum()

def read_feather_fn(fn):
    return feather.read_dataframe(str(fn))

def flatten_and_cat(conts, extra: List[pd.DataFrame], cats=None, sl=30):
    columns = list(conts[0].columns)
    cyc_cont = pd.DataFrame([o.values.flatten() for o in conts])

    # add dependent variables
    cyc_cont = pd.concat([cyc_cont, *extra], axis=1, copy=False)
    columns = ['_'.join([str(o), str(i)]) for i in range(sl) for o in columns] # matrix
    columns += [i for o in extra for i in o.columns.tolist()]
    cyc_cont.columns = columns

    print('cyc_cont.shape', cyc_cont.shape)
    return cyc_cont

def extract_input(df, idx, sl, cont_names=None):
    # idx = list(chain(*[list(range(df.index.get_loc(i).start+idx[i], df.index.get_loc(i).start+idx[i]+sl)) 
        # for i in df.index.levels[0]]))
    idx = [list(range(df.index.get_loc(i).start+idx[i], df.index.get_loc(i).start+idx[i]+sl)) 
            for i in df.index.levels[0] if i in df.index]
    assert(max(len(o) for o in idx) == sl)
    assert(min(o[0] for o in idx) >= 0)
    idx = list(chain(*idx))

    df = df.iloc[idx]
    if cont_names is not None:
        df = df.loc[:,cont_names]
    return df

def normalize_df(df, mean, std):
    return (df.loc[:,mean.index] - mean) / (eps + std)

def normalize_a(a, mean, std):
    return (a - mean) / (eps + std)

def denormalize(y, mean, std):
    if isinstance(y, torch.Tensor): 
        stats = (mean, std)
        stats = (tensor(o).float() for o in stats)
        if y.device.type == 'cuda': stats = (o.cuda() for o in stats)
        mean, std = stats
    return y * (eps + std) + mean

def concat_cycles(cycles):
    return pd.concat(cycles, axis=0, keys=range_of(cycles), names=['cycle', 't'])

def ni(it): return next(iter(it))

def my_loss_batch(model, xb, yb, cb_handler=None):
    out = model(*xb)
    out = cb_handler.on_loss_begin(out)
    return out.cpu()

def valid(learner, context, ds_type=DatasetType.Valid, ret_grad=False):
    xb,yb = learner.data.one_batch(ds_type, detach=False, denorm=False)
    xb[1].requires_grad = True # enable the calculation of gradient
    yb = to_detach(yb)

    # Get prediction
    cb_handler = CallbackHandler(learner.callbacks) # rnn trainer
    learner.model.eval().reset()
    # set rnn to train mode to obtain gradient with respect to x
    apply_leaf(learner.model, lambda m: m.train() if m.__module__.endswith('rnn') else None)
    pb = my_loss_batch(learner.model, xb, yb, cb_handler=cb_handler)

    # Calculate scores
    loss, mapd = learner.loss_func(pb, yb), learner.metrics[0](pb, yb)
    if 1:
        print('Predicting mean: mapd =', to_np(learner.metrics[0](yb, yb.mean(0))))
    print('mapd', to_np(mapd))

    # Plot p against y. Expect linearity
    x_np, y_np, p_np = (to_np(o).squeeze() for o in (xb[1],yb,pb))
    yb_denormed, pb_denormed = (denormalize(o, *context.stat_y) for o in (y_np, p_np))
    # xb_denormed = denormalize(x_np, *context.stat_x)
    y_df, p_df = (pd.DataFrame(d, columns=[l+postfix for l in context.dep_var]) for d, postfix in zip((yb_denormed, pb_denormed), ('_y', '_p')))
    result = pd.concat([y_df, p_df], axis=1)

    for x, y in zip(result.columns, result.columns[result.columns.shape[0]//2:]):
        ax = sns.jointplot(x, y, result, kind='reg', ratio=9, height=8)
        set_ax_font(ax.ax_joint)

    if ret_grad:
        # Show influence of columns by taking derivatives
        grad = to_np(torch.autograd.grad(mapd, xb[1], retain_graph=True)[0])
        # Reshape to reflect sl dimension and average along that dimension
        grad = grad.reshape((grad.shape[0], context.sl,
            context.n_cont))
        grad = np.abs(grad).mean(0).mean(0)

        cont_names = [s[:-2] for s in context.cyc_cont.columns[:context.n_cont]]
        influence = pd.DataFrame(grad, index=cont_names, columns=['grad']).sort_values('grad', ascending=False)
        # plot the most influential columns
        # x_df = reconstruct_flattened(xb_denormed, context)
    else:
        influence = None

    return result, influence

def predict_test(model, context, is_task1=True):
    xb = read_test_files(is_task1)

def read_test_files(is_task1=True):
    pass


def reconstruct_flattened(x, context):
    if isinstance(x, Tensor): x = to_np(x)
    x = x.reshape(context.sl, len(context.cont_names))
    return pd.DataFrame(x, columns=context.cont_names)
    



