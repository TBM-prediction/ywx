from utils.misc import *
from config import dep_var1, dep_var2

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

    def cycles1(self, fn=None, first_idx=0, min_cycle_length=500, debug=False):
        # split time series into cycles
        df = self.df_raw.loc[:,'推进速度']
        begins, ends = zero_boundary(df)
        begins, ends = ends[:-1], begins[1:]
        print('begins', begins)
        print('ends', ends)

        self.cycles = []
        self.short_cycles = []
        for b, e in zip(begins, ends):
            # filter out cycles that are too short
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
        if debug: return self.cycles, self.short_cycles

    def cycles2(self, save_fn=None, first_idx=0):
        # split time series into cycles
        # use multi-index dataframe
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

    def drop_redundent_columns(self):
        cycles = self.cycles # shorten notation
        cycles.drop(columns=['运行时间'], inplace=True, errors='ignore')

        # Drop constant columns (std == 0)
        to_drop = set()
        to_drop |= set(cycles.columns[(cycles.std() == 0)])
        to_drop |= {'主驱动2#电机输出功率', '主驱动6#电机输出功率', '主驱动8#电机输出功率', '主驱动2#电机电流', '主驱动5#电机电流', '主驱动8#电机电流', '主驱动2#电机扭矩', '主驱动6#电机扭矩', '主驱动8#电机扭矩', '主驱动2#电机输出频率', '主驱动6#电机输出频率', '主驱动8#电机输出频率',
                     '减速机5#温度', '减速机4#温度', '减速机9#温度', '减速机3#温度', '减速机2#温度', '减速机7#温度', '减速机6#温度', '减速机8#温度', # drop anything other than 10# to be consistant with above
                     '主驱动3#电机输出频率', '主驱动3#电机输出功率', '主驱动3#电机扭矩',
                     # drop 7#, 5#, preserving 9
                     '主驱动7#电机扭矩', '主驱动5#电机扭矩', '主驱动7#电机输出功率', '主驱动5#电机输出功率', '主驱动7#电机电流', '主驱动5#电机电流', '主驱动7#电机输出频率', '主驱动5#电机输出频率',
                     '推进速度.1', '推进给定速度百分比', '右推进油缸行程检测',
                     '时间戳', '刀盘运行时间',
                     '二次风机频率设置', '推进压力', '刀盘给定转速显示值',
                     '刀盘转速电位器设定值', '变频柜回水温度报警值',
                     '拖车尾部CH4浓度', '污水箱压力检测'}
        to_drop = [c for c in to_drop if c in cycles.columns] + ['运行时间']
        print('valid to_drop', to_drop, len(to_drop))
        cycles.drop(columns=to_drop, inplace=True, errors='ignore')
        return to_drop

        # # Before dropping numbered columns, average them
        # prog = re.compile('.*\d*#.*')
        # average_columns = [s for s in cycles.columns 
                # if re.match(prog, s) is not None]
        # # If we print the names of the columns, we can see the names are in
            # # groups of 10
        # n_dropped = 0
        # for i in range(0, len(average_columns), 10):
            # # Group columns of similar sensors
            # g = average_columns[i:i+10]
            # new_column = re.sub('\d#', '', g[0])
            # # Replace the first column in the group with the mean of the group
            # cycles.loc[:,g[0]] = cycles.loc[:,g].mean(1)
            # # Drop the rest of the group
            # n_dropped += len(g) - 1
            # cycles.drop(columns=g[1:], inplace=True)
            # # Remove the number in the name of the first column in the group
            # cycles.rename(columns={g[0]: new_column}, inplace=True, copy=False)
        # print('Replace numbered sensors with mean:', n_dropped)

        # # Drop columns of high correlation:
        # # https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/
        # # Protect '桩号' because 用来训练地质预测模型. Remember to exclude this column from input data.
        # protected_columns = ['桩号'] + dep_var1 + dep_var2
        # # Think about how to protect certain columns from being dropped. If
        # # columns i and j (i < j) are highly correlated, we only drop one of
        # # them by masking the correlation matrix with it's upper triangle, where
        # # the first index is smaller.

        # # Thus we only preserve correlation [i, j] in the masked matrix. And because
        # # we drop columns according to column number, column i is protected 
        # # from being dropped.

        # # In conclusion, to protect certain columns from being dropped, we
        # # should move it forward before computing the correlation matrix to
        # # assign it a smaller index number (which is i in the analysis).
        # reordered_columns = protected_columns + [c for c in cycles.columns if c not in protected_columns]
        # cycles = cycles.loc[:, reordered_columns]

	# # Create correlation matrix
        # corr_matrix = cycles.corr().abs()
        # # Select upper triangle of correlation matrix
        # upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        # # Find index of feature columns with correlation greater than 0.95
        # to_drop = [c for c in upper.columns if np.any(upper.loc[:,c] > 0.95)]
        # assert all(o not in protected_columns for o in to_drop), """A protected
        # column is dropped. This happens when at least two of the protected
        # columns have high correlation."""
        # # Drop
        # n_dropped = len(to_drop); print('Drop columns of high correlation:',
                # n_dropped)
        # cycles.drop(columns=to_drop, inplace=True)


    def get_x(self, noise_size=(-3,4), normalize=True, stat_x=None):
        df = self.cycles

        # Exclude '桩号' from input data
        cont_names = [c for c in df.columns if c != '桩号']
        # cont_names = df.columns # include it
        print('cont names', len(cont_names))

        if normalize:
            norm_cont_names = cont_names + ['桩号']
            tile = extract_input(df, self.idx, self.context.sl, norm_cont_names)
            mean,std = ifnone(stat_x,
                    (tile.loc[:,norm_cont_names].mean(),tile.loc[:,norm_cont_names].std()))
            df.loc[:,cont_names] = normalize_df(df, mean.iloc[:-1], std.iloc[:-1])

        tile = extract_input(df, self.idx, self.context.sl, cont_names) # extract with normalized df
        tiles = [tile]

        # Tile *mulr* times with noise
        if self.context.mulr > 1:
            # m, M = noise_size
            # noises = (np.random.random(self.context.mulr-1) * (M-m+1)).astype('uint8') + m
            noises = np.random.permutation(range(*noise_size))[:self.context.mulr-1]
            print('Noises:', noises)
            tiles += [extract_input(df, self.idx+n, self.context.sl, cont_names) for n in tqdm_notebook(noises, 'tile_with_noise')]

        # flatten along cycle
        tiles = [t.loc[i] for t in tiles for i in t.index.levels[0] if i in t.index]
        if normalize:
            return tiles, (mean, std)
        else:
            return tiles

    def get_extra_x_task2(self, cycles=None):
        cycles = ifnone(cycles, self.cycles)
        target_names = ['桩号', '推进速度', '刀盘转速']
        target_columns = [cycles.loc[o,target_names] for o in cycles.index.levels[0]]

        # Use the mean of the values at 稳定段
        # TODO: think again
        y = pd.DataFrame([o.iloc[200:-100].mean(0) for o in target_columns], columns=target_names)

        if self.context.mulr > 1:
            y = pd.concat([y]*self.context.mulr).reset_index(drop=True)

        return y


    def get_y(self, cycles=None):
        cycles = ifnone(cycles, self.cycles)
        target_names = self.context.dep_var
        target_columns = [cycles.loc[o,target_names] for o in cycles.index.levels[0]]

        if target_names == ['桩号']:
            # Use the value at the beginning of 上升段
            y = pd.DataFrame([cycle.iloc[start_idx] for cycle,start_idx in
                zip(target_columns,self.idx)],
                columns=target_names).reset_index(drop=True)
        else:
            # Use the mean of the values at 稳定段
            y = pd.DataFrame([o.iloc[200:-100].mean(0) for o in target_columns], columns=target_names)

        if self.context.mulr > 1:
            y = pd.concat([y]*self.context.mulr).reset_index(drop=True)

        return y

