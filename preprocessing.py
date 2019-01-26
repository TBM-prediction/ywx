from utils.misc import *

class DataFormatter:
    def __init__(self, fns):
        if not isinstance(fns, list): fns = [fns]
        self.df_raw = [pd.read_csv(o, sep='\t', index_col=False, low_memory=False, parse_dates=['运行时间']) 
                for o in fns]
        self.df_raw = pd.concat(self.df_raw)

    def remove_noise(self):
        self.df_raw[self.df_raw['推进速度'] > 300] = 0

    def remove_anomaly(self, min_num_zeros = 5):
        df = self.df_raw['推进速度']
        begins, ends, num_zeros = zero_boundary(df)

        # interpolate on continuous columns
        for b, e, l in zip(begins, ends, num_zeros):
            if l <= min_num_zeros:
                self.df_raw.iloc[b:e,2:] = np.tile(((self.df_raw.iloc[b-1,2:] + self.df_raw.iloc[e,2:])/2).values, (l, 1))


    def cycles1(self):
        # split time series into cycles by speed
        df = self.df_raw['推进速度']
        begins, ends, _ = zero_boundary(df)
        begins, ends = ends[:-1], begins[1:]
        print('begins', begins)
        print('ends', ends)

        self.cycles = []
        for b, e in zip(begins, ends):
            min_cycle_length = 500
            # fileter out cycles that are too short
            if e - b > min_cycle_length:
                self.cycles.append(self.df_raw.iloc[b:e])

    def stages1(self):
        # split cycles into 
        # 1. 空推段, 2. 上升段, 3. 稳定段 and 4. 稳定段平均值
        input_columns, target_columns = zip(*[self.get_columns(o) for o in self.cycles])

        # self.stages = 

    @classmethod
    def get_columns(df, is_problem1=True):
        target_names = ['推进速度电位器设定值', '刀盘转速电位器设定值'] if is_problem1 else ['总推进力', '刀盘扭矩']
        input_names = [o for o in df.columns if o not in target_names]

        return (df[input_names], df[target_names])

    # @classmethod
    # def get_model_data(

