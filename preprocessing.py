from utils.misc import *

class DataFormatter:
    def __init__(self, csv_fns=None, cycle_feathers=None):
        if csv_fns is not None:
            fns = csv_fns
            if not isinstance(fns, list): fns = [fns]
            with concurrent.futures.ThreadPoolExecutor() as e:
                f = partial(pd.read_csv, sep='\t', index_col=False, low_memory=False, parse_dates=['运行时间'])
                df_raw = list(tqdm_notebook(e.map(f, fns), total=len(fns)))
            self.df_raw = pd.concat(df_raw, copy=False)

        elif cycle_feathers is not None:
            fns = cycle_feathers
            if not isinstance(fns, list): fns = [fns]
            with concurrent.futures.ThreadPoolExecutor() as e:
                self.cycles = list(tqdm_notebook(e.map(read_feather_fn, fns), 
                    desc='read_dataframe', total=len(fns)))
        else:
            print('wrong arguments')


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
            cyc = self.df_raw.iloc[b:e].reset_index(drop=True)
            if e - b > min_cycle_length:
                self.cycles.append(cyc)
            else:
                self.short_cycles.append(cyc)
        if fn is not None:
            for i, cyc in enumerate(self.cycles):
                cyc.to_feather(str(fn) + str(first_idx+i))
        print(f'got {len(self.cycles)} cycles, filtered {len(self.short_cycles)} short cycles.')
        #print([len(o) for o in self.short_cycles])

    def get_y(self, cycles=None, is_problem1=True):
        cycles = cycles or self.cycles

        target_names = ['推进速度电位器设定值', '刀盘转速电位器设定值'] if is_problem1 else ['总推进力', '刀盘扭矩']
        if is_problem1:
            target_columns = [o.loc[:,target_names] for o in cycles]

            # TODO: go more sophisticated
            # take the last point and mode of values respectively
            y = pd.DataFrame([(o.iloc[-1,0], o.iloc[:,1].mode().values[0]) for o in target_columns], columns=target_names)
        else:
            raise NotImplementedError

        return y

