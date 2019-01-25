from utils.mini_imports import *

class DataFormatter:
    def __init__(self, fns):
        if not isinstance(fns, list): fns = [fns]
        self.df_raw = [pd.read_csv(o, sep='\t', index_col=False, low_memory=False, parse_dates=['运行时间']) 
                for o in fns]
        self.df_raw = pd.concat(self.df_raw)

    def remove_noise(self):
        self.df_raw[self.df_raw['推进速度'] > 300] = 0

    def remove_anomaly(self, min_num_zeros = 5):
        anomaly = self.df_raw['推进速度']
        zeros = (anomaly == 0).astype('int')
        boundary = zeros.diff()

        # handle boundary condition (no pun intended)
        if zeros.iloc[0] == 1: boundary.iloc[0] = 1
        if zeros.iloc[-1] == 1: boundary.iloc[-1] = -1

        begins = np.argwhere(np.array((boundary == 1))).squeeze()
        ends = np.argwhere(np.array((boundary == -1))).squeeze()
        num_zeros = [e - b for b, e in zip(begins, ends)]

        # interpolate on continuous columns
        for b, e, l in zip(begins, ends, num_zeros):
            if l <= min_num_zeros:
                self.df_raw.iloc[b:e,2:] = np.tile(((self.df_raw.iloc[b-1,2:] + self.df_raw.iloc[e,2:])/2).values, (l, 1))
                print(self.df_raw['推进速度'].iloc[b:e])

                


    def split_cycles1(self):
        pass
    
