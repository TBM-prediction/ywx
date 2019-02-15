from utils.misc import *

class DataFormatter:
    def __init__(self, context, csv_fns=None, cycle_feathers=None):
        self.context = context
        if csv_fns is not None:
            fns = csv_fns
            if not isinstance(fns, list): fns = [fns]
            with concurrent.futures.ThreadPoolExecutor(4) as e:
                f = partial(pd.read_csv, sep='\t', index_col=False, low_memory=False, parse_dates=['运行时间'])
                df_raw = list(tqdm_notebook(e.map(f, fns), total=len(fns)))
            self.df_raw = pd.concat(df_raw, copy=False)

        elif cycle_feathers is not None:
            fns = cycle_feathers
            if not isinstance(fns, list): fns = [fns]
            with concurrent.futures.ThreadPoolExecutor() as e:
                cycles = list(tqdm_notebook(e.map(read_feather_fn, fns), 
                    desc='read_dataframe', total=len(fns)))
            self.cycles = concat_cycles(cycles)

        else:
            print('wrong arguments')


    def cleanup(self, min_num_zeros = 5):
        valid_columns = [o for o in self.df_raw.columns if o not in ('EP2次数设置', '刀盘喷水增压泵压力')]
        self.df_raw = self.df_raw.loc[:,valid_columns]
        df = self.df_raw.loc[:,'推进速度']
        self.df_raw.loc[df > 300] = 0
        begins, ends = zero_boundary(df)
        print(f'got {len(begins)} zeros segments')

        idx = [o for b, e in zip(begins, ends) if e - b <= min_num_zeros
                for o in range(b, e)]
        self.df_raw[self.df_raw.index.isin(idx)] = np.NaN
        
        print('Interpolating...')
        self.df_raw.interpolate(inplace=True)

    def cycles1(self, fn=None, first_idx=0):
        # split time series into cycles
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

    def cycles2(self, save_fn=None, first_idx=0):
        # split time series into cycles
        df = self.df_raw.loc[:,'推进速度']
        begins, ends = zero_boundary(df)
        begins, ends = ends[:-1], begins[1:]
        print('begins', begins)
        print('ends', ends)

        cycles = []
        short_cycles = 0
        for b, e in zip(begins, ends):
            min_cycle_length = 500
            # fileter out cycles that are too short
            if e - b > min_cycle_length:
                cycles.append(self.df_raw.iloc[b:e].reset_index(drop=True))
            else:
                short_cycles += 1
        
        # concatenate cycles to a multi-indexed df
        self.cycles = concat_cycles(cycles)

        if save_fn is not None:
            for i, cyc in enumerate(cycles):
                cyc.to_feather(str(fn) + str(first_idx+i))
        print(f'got {len(self.cycles)} cycles, filtered {len(self.short_cycles)} short cycles.')


    def beginning_index(self, dfs=None, postpond=0, thresh=100):
        if dfs is None: 
            dfs = self.cycles
        else:
            if not isinstance(dfs, collections): 
                dfs = [dfs]
        self.idx = np.array([(rolling_cumsum(dfs.loc[i]) > thresh).idxmax() + postpond 
                    for i in tqdm_notebook(dfs.index.levels[0], 'beginning_index')])
        return self.idx

    def get_x(self, normalize=True):
        df = self.cycles
        cont_names = self.context.cont_names

        if normalize:
            tile = extract_input(df, self.idx, self.context.sl, cont_names)
            mean,std = tile.loc[:,cont_names].mean(),tile.loc[:,cont_names].std()
            df.loc[:,cont_names] = normalize_df(df, mean, std)

        tile = extract_input(df, self.idx, self.context.sl, cont_names) # extract with normalized df
        tiles = [tile]

        if self.context.mulr > 1:
            m, M = noise_size
            noises = (np.random.random(self.context.mulr-1) * (M-m+1)).astype('uint8') + m
            tiles += [extract_input(df, self.idx+n, self.context.sl, cont_names) for n in tqdm_notebook(noises, 'tile_with_noise')]

        tiles = [t.loc[i] for t in tiles
                            for i in t.index.levels[0]]
        if normalize:
            return tiles, (mean, std)
        else:
            return tiles


    def get_y(self, cycles=None, normalize=True):
        cycles = ifnone(cycles, self.cycles)
        target_names = self.context.dep_var
        target_columns = [cycles.loc[o,target_names] for o in cycles.index.levels[0]]

        # TODO: go more sophisticated
        # take the last point and mode of values respectively
        cols = [o.astype('int') for o in target_columns]
        y = pd.DataFrame([(o.iloc[200:,0].mode().values[0].astype('float'), o.iloc[:,1].mode().values[0].astype('float')) for o in cols], columns=target_names)

        if normalize:
            mean,std = y.mean(),y.std()
            y = normalize_df(y, mean, std)
        if self.context.mulr > 1:
            y = pd.concat([y]*self.context.mulr).reset_index(drop=True)

        return y, (mean, std) if normalize else y

# before using context
# class DataFormatter:
#     def __init__(self, context=None, csv_fns=None, cycle_feathers=None):
#         self.context = context
#         if csv_fns is not None:
#             fns = csv_fns
#             if not isinstance(fns, list): fns = [fns]
#             with concurrent.futures.ThreadPoolExecutor(4) as e:
#                 f = partial(pd.read_csv, sep='\t', index_col=False, low_memory=False, parse_dates=['运行时间'])
#                 df_raw = list(tqdm_notebook(e.map(f, fns), total=len(fns)))
#             self.df_raw = pd.concat(df_raw, copy=False)
# 
#         elif cycle_feathers is not None:
#             fns = cycle_feathers
#             if not isinstance(fns, list): fns = [fns]
#             with concurrent.futures.ThreadPoolExecutor() as e:
#                 cycles = list(tqdm_notebook(e.map(read_feather_fn, fns), 
#                     desc='read_dataframe', total=len(fns)))
#             self.cycles = concat_cycles(cycles)
# 
#         else:
#             print('wrong arguments')
# 
# 
#     def cleanup(self, min_num_zeros = 5):
#         valid_columns = [o for o in self.df_raw.columns if o not in ('EP2次数设置', '刀盘喷水增压泵压力')]
#         self.df_raw = self.df_raw.loc[:,valid_columns]
#         df = self.df_raw.loc[:,'推进速度']
#         self.df_raw.loc[df > 300] = 0
#         begins, ends = zero_boundary(df)
#         print(f'got {len(begins)} zeros segments')
# 
#         idx = [o for b, e in zip(begins, ends) if e - b <= min_num_zeros
#                 for o in range(b, e)]
#         self.df_raw[self.df_raw.index.isin(idx)] = np.NaN
#         
#         print('Interpolating...')
#         self.df_raw.interpolate(inplace=True)
# 
#     def cycles1(self, fn=None, first_idx=0):
#         # split time series into cycles
#         df = self.df_raw.loc[:,'推进速度']
#         begins, ends = zero_boundary(df)
#         begins, ends = ends[:-1], begins[1:]
#         print('begins', begins)
#         print('ends', ends)
# 
#         self.cycles = []
#         self.short_cycles = []
#         for b, e in zip(begins, ends):
#             min_cycle_length = 500
#             # fileter out cycles that are too short
#             cyc = self.df_raw.iloc[b:e].reset_index(drop=True)
#             if e - b > min_cycle_length:
#                 self.cycles.append(cyc)
#             else:
#                 self.short_cycles.append(cyc)
#         if fn is not None:
#             for i, cyc in enumerate(self.cycles):
#                 cyc.to_feather(str(fn) + str(first_idx+i))
#         print(f'got {len(self.cycles)} cycles, filtered {len(self.short_cycles)} short cycles.')
#         #print([len(o) for o in self.short_cycles])
# 
#     def cycles2(self, save_fn=None, first_idx=0):
#         # split time series into cycles
#         df = self.df_raw.loc[:,'推进速度']
#         begins, ends = zero_boundary(df)
#         begins, ends = ends[:-1], begins[1:]
#         print('begins', begins)
#         print('ends', ends)
# 
#         cycles = []
#         short_cycles = 0
#         for b, e in zip(begins, ends):
#             min_cycle_length = 500
#             # fileter out cycles that are too short
#             if e - b > min_cycle_length:
#                 cycles.append(self.df_raw.iloc[b:e].reset_index(drop=True))
#             else:
#                 short_cycles += 1
#         
#         # concatenate cycles to a multi-indexed df
#         self.cycles = concat_cycles(cycles)
# 
#         if save_fn is not None:
#             for i, cyc in enumerate(cycles):
#                 cyc.to_feather(str(fn) + str(first_idx+i))
#         print(f'got {len(self.cycles)} cycles, filtered {len(self.short_cycles)} short cycles.')
# 
# 
#     def beginning_index(self, dfs=None, postpond=0, thresh=100):
#         if dfs is None: 
#             dfs = self.cycles
#         else:
#             if not isinstance(dfs, collections): 
#                 dfs = [dfs]
#         self.idx = np.array([(rolling_cumsum(dfs.loc[i]) > thresh).idxmax() + postpond 
#                     for i in tqdm_notebook(dfs.index.levels[0], 'beginning_index')])
#         return self.idx
# 
#     def get_x(self, context, noise_size=(-2, 5), normalize=True):
#         df,mulr,cont_names,sl = self.cycles,context.mulr,context.cont_names,context.sl
#         
# 
#         if normalize:
#             tile = extract_input(df, self.idx, config.sl, cont_names)
#             mean,std = tile.loc[:,cont_names].mean(),tile.loc[:,cont_names].std()
#             df.loc[:,cont_names] = normalize_df(df, mean, std)
# 
#         tile = extract_input(df, self.idx, sl, cont_names) # extract with normalized df
#         tiles = [tile]
# 
#         if mulr > 1:
#             m, M = noise_size
#             noises = (np.random.random(config.mulr-1) * (M-m+1)).astype('uint8') + m
#             tiles += [extract_input(df, self.idx+n, config.sl, cont_names) for n in tqdm_notebook(noises, 'tile_with_noise')]
# 
#         tiles = [t.loc[i] for t in tiles
#                             for i in t.index.levels[0]]
#         if normalize:
#             return tiles, (mean, std)
#         else:
#             return tiles
# 
# 
#     def get_y(self, dep_vars, cycles=None, normalize=True):
#         cycles = ifnone(cycles, self.cycles)
#         target_names = dep_vars
#         target_columns = [cycles.loc[o,target_names] for o in cycles.index.levels[0]]
# 
#         # TODO: go more sophisticated
#         # take the last point and mode of values respectively
#         cols = [o.astype('int') for o in target_columns]
#         y = pd.DataFrame([(o.iloc[200:,0].mode().values[0].astype('float'), o.iloc[:,1].mode().values[0].astype('float')) for o in cols], columns=target_names)
# 
#         if normalize:
#             mean,std = y.mean(),y.std()
#             y = normalize_df(y, mean, std)
#             return y, (mean, std)
#         else: 
#             return y
# 
