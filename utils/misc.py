from .imports import *

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
    diff[diff<0] = 0
    return diff.cumsum()

def beginning_index(df):
    cs = rolling_cumsum(df)
    cs = cs > 100
    return cs.idxmax()

def read_feather_fn(fn):
    return feather.read_dataframe(str(fn))

def flatten_and_cat(conts, deps, cats=None):
    columns = list(conts[0].columns)
    cyc_cont = pd.DataFrame([o.values.flatten() for o in conts])

    # add dependent variables
    cyc_cont = pd.concat([cyc_cont, deps], axis=1, copy=False)
    cyc_cont.columns = ['_'.join([str(o), str(i)]) for i in range(30)
                                                    for o in columns] + list(deps.columns)

    print('cyc_cont.shape', cyc_cont.shape)
    return cyc_cont

def tile_with_noise(cycles, idx, mulr, cont_names, noise_size=(-2, 5)):
    noise = (np.random.random(mulr) * (noise_size[1]-noise_size[0]+1)).astype('uint8') + noise_size[0]
    df_conts = [o.loc[:,cont_names].iloc[i+n:i+30+n] for n in tqdm_notebook(noise)
                                                    for i, o in zip(idx, cycles)]
    return df_conts


l1 = nn.L1Loss(reduction='none')
def our_metrics(input, target):
    # score range: (-NaN, 2]
    return 2 - (l1(input, target) / target).sum() / target.view(-1).size()[0]

