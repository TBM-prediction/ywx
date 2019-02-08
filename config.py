from utils.misc import *
from preprocessing import *
from mymodels import *
from databunch import *

# import jtplot submodule from jupyterthemes
# currently installed theme will be used to
# set plot style if no arguments provided
from jupyterthemes import jtplot
jtplot.style()

# pandas ipython output bug workaround. Fixed on pandas 0.24.1
# get_ipython().config.get('IPKernelApp', {})['parent_appname'] = ""


class Config:

    data_path = Path('tbmData/data')
    fn_txt = sorted(data_path.glob('*.txt'))
    fn_cycles = Path('tbmData/cycles1')

    def __init__(self, debug=True, gpu_start=2, sl=30, selected_columns=False, is_problem1=True, valid_ratio=0.2):
        self.debug,self.gpu_start,self.sl,self.selected_columns = debug,gpu_start,sl,selected_columns
        self.mulr = mulr = 50 if debug else 7
        if selected_columns:
            self.cont_names = ['推进速度', '主驱动1#电机扭矩', '刀盘功率', '刀盘扭矩', '刀盘转速','主液压油箱温度', '前点偏差X', '总推进力']
            self.n_cont = len(self.cont_names)
        else:
            self.n_cont = 5 if debug else 192
        num_cycles = self.num_cycles = 5 if debug else 3481
        
        train_ratio = 1 - valid_ratio
        self.train_idx = train_idx = np.arange(int(num_cycles * valid_ratio), num_cycles)
        self.valid_idx = valid_idx = np.arange(int(num_cycles * valid_ratio))
        self.train_idx_tile = (train_idx[:, None] + np.arange(mulr) * num_cycles).flatten()
        self.valid_idx_tile = (valid_idx[:, None] + np.arange(mulr) * num_cycles ).flatten()  # take from all tiles

        self.bs = int(num_cycles * train_ratio)
        self.sl = sl
        
        if is_problem1:
            self.dep_var = ['推进速度电位器设定值', '刀盘转速电位器设定值']
        else:
            raise NotImplementedError


