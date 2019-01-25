from utils.mini_imports import *

class DataFormatter:
    def __init__(self, fns):
        if not isinstance(fns, list): fns = [fns]
        self.df_raw = [pd.read_csv(o, sep='\t', index_col=False, low_memory=False, parse_dates=['运行时间']) 
                for o in fns]
        self.df_raw = pd.concat(self.df_raw)

    def remove_noise(self):
        self.df_raw[self.df_raw['推进速度'] > 300] = 0

    def split_cycles1(self):
        pass
    
