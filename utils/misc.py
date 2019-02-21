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
    df = df.loc[:,'刀盘功率'].rolling(10, center=True).mean()
    diff = df.diff()
    diff.loc[diff<0] = 0
    return diff.cumsum()

def read_feather_fn(fn):
    return feather.read_dataframe(str(fn))

def flatten_and_cat(conts, deps, cats=None, sl=30):
    columns = list(conts[0].columns)
    cyc_cont = pd.DataFrame([o.values.flatten() for o in conts])

    # add dependent variables
    cyc_cont = pd.concat([cyc_cont, deps], axis=1, copy=False)
    cyc_cont.columns = ['_'.join([str(o), str(i)]) for i in range(sl) for o in columns] + deps.columns.tolist()

    print('cyc_cont.shape', cyc_cont.shape)
    return cyc_cont

def extract_input(df, idx, sl, cont_names=None):
    idx = list(chain(*[list(range(df.index.get_loc(i).start+idx[i], df.index.get_loc(i).start+idx[i]+sl)) 
        for i in df.index.levels[0]]))
    df = df.iloc[idx]
    if cont_names is not None:
        df = df.loc[:,cont_names]
    return df

def normalize_df(df, mean, std):
    return (df.loc[:,mean.index] - mean) / (eps + std)

def denormalize(y, mean, std):
    stats = (mean, std)
    if isinstance(y, torch.Tensor): 
        stats = (tensor(o).float() for o in stats)
        if y.device.type == 'cuda': stats = (o.cuda() for o in stats)
    mean, std = (o for o in stats)
    return y * (eps + std) + mean

# def tile_with_noise(df, idx, config, noise_size=(-2, 5), normalize=True):
    # mulr, cont_names, sl = config.mulr, config.cont_names, config.sl

    # if normalize:
        # tile = extract_input(df, idx, config.sl, cont_names)
        # mean,std = tile.loc[:,cont_names].mean(),tile.loc[:,cont_names].std()
        # df.loc[:,cont_names] = normalize_df(df, mean, std)

    # tile = extract_input(df, idx, config.sl, cont_names) # extract with normalized df
    # tiles = [tile]

    # if config.mulr > 1:
        # m, M = noise_size
        # noises = (np.random.random(config.mulr-1) * (M-m+1)).astype('uint8') + m
        # tiles += [extract_input(df, idx+n, config.sl, cont_names) for n in tqdm_notebook(noises, 'tile_with_noise')]

    # tiles = [t.loc[i] for t in tiles
                        # for i in t.index.levels[0]]
    # if normalize:
        # return tiles, (mean, std)
    # else:
        # return tiles

def concat_cycles(cycles):
    return pd.concat(cycles, axis=0, keys=range_of(cycles), names=['cycle', 't'])


def ni(it): return next(iter(it))

def train_eval(learn):
    # print score
    pass

def evaluate(learn):
    pass
