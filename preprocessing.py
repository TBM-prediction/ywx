from utils.misc import *

class DataFormatter:
    def __init__(self, fns):
        if not isinstance(fns, list): fns = [fns]
        with concurrent.futures.ThreadPoolExecutor() as e:
            f = partial(pd.read_csv, sep='\t', index_col=False, low_memory=False, parse_dates=['运行时间'])
            df_raw = list(tqdm_notebook(e.map(f, fns), total=len(fns)))
        self.df_raw = pd.concat(df_raw, copy=False)

    def cleanup(self, min_num_zeros = 5):
        valid_columns = [o for o in self.df_raw.columns if o not in ('EP2次数设置', '刀盘喷水增压泵压力')]
        self.df_raw = self.df_raw.loc[:,valid_columns]
        df = self.df_raw.loc[:,'推进速度']
        self.df_raw.loc[df > 300] = 0
        begins, ends = zero_boundary(df)
        print(f'got {len(begins)} zeros segments')

        # interpolate on continuous columns
       #def f(b, e):
       #    if e - b <= min_num_zeros:
       #        self.df_raw.iloc[b:e] = np.NaN
       #        #self.df_raw.iloc[b:e,2:] = np.tile(((self.df_raw.iloc[b-1,2:] + self.df_raw.iloc[e,2:])/2).values, (l, 1))

       #with concurrent.futures.ThreadPoolExecutor(defaults.cpus//2) as e:
       #    list(tqdm_notebook(e.map(f, begins, ends)))
        idx = [o for b, e in zip(begins, ends) if e - b <= min_num_zeros
                for o in range(b, e)]
        self.df_raw[self.df_raw.index.isin(idx)] = np.NaN
        
        print('Interpolating...')
        self.df_raw.interpolate(inplace=True)

    def cycles1(self, fn=None, first_idx=0):
        # split time series into cycles by speed
        df = self.df_raw.loc[:,'推进速度']
        begins, ends = zero_boundary(df)
        begins, ends = ends[:-1], begins[1:]
        print('begins', begins)
        print('ends', ends)

        self.cycles = []
        self.short_cycles = []
        for b, e in zip(begins, ends):
            min_cycle_length = 500
            # fileter out cycles that are too short
            if e - b > min_cycle_length:
                self.cycles.append(self.df_raw.iloc[b:e])
            else:
                self.short_cycles.append(self.df_raw.iloc[b:e])
        if fn is not None:
            for i, cyc in enumerate(self.cycles):
                cyc.reset_index(drop=True).to_feather(str(fn) + str(first_idx+i))
        print(f'got {len(self.cycles)} cycles, filtered {len(self.short_cycles)} short cycles.')
        #print([len(o) for o in self.short_cycles])

    def stages1(self):
        # split cycles into 
        # 1. 空推段, 2. 上升段, 3. 稳定段 and 4. 稳定段平均值
        pass

    def get_model_data(self, is_problem1=True):
        if is_problem1:
            input_columns, target_columns = zip(*[self.get_columns(o, is_problem1) for o in self.cycles])
            x = [o.values for o in input_columns]
            y = [(o.iloc[-1,0], o.iloc[:,1].mode().values[0]) for o in target_columns] # TODO: go more sophisticated

        return x, y
            
    @classmethod
    def get_columns(cls, df, is_problem1=True):
        target_names = ['推进速度电位器设定值', '刀盘转速电位器设定值'] if is_problem1 else ['总推进力', '刀盘扭矩']
        input_names = [o for o in df.columns if o not in target_names]

        return (df[input_names], df[target_names])

def rolling_cumsum(df):
    df = df['刀盘功率'].rolling(10, center=True).mean()
    diff = df.diff()
    diff[diff<0] = 0
    return diff.cumsum()

def beginning_index(df):
    cs = rolling_cumsum(df)
    cs = cs > 100
    return cs.idxmax()

