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
    params = []
    for target_param in ('推进速度', '刀盘转速', '推进力', '扭矩'):
        for param in columns_list:
            if target_param in param:
                params.append(param)
#     if merge:
#         import re
#         merge_params = [o for o in params if re.match('主驱动\d#电机扭矩', o)]
#         df = pd.concat([df, df[merge_params].mean(axis=1).copy()], axis=1, copy=False)
    if print_:
        print(params)
    return df.loc[list(params)]

