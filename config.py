dep_var1 = ['推进速度', '刀盘转速']
dep_var2 = ['总推进力', '刀盘扭矩']

from utils.misc import *
from preprocessing import *
from mymodels import *
from databunch import *
from crits import *

# import jtplot submodule from jupyterthemes
# currently installed theme will be used to
# set plot style if no arguments provided
from jupyterthemes import jtplot; jtplot.style()

from typing import Callable


@dataclass
class Context:

    exp_name: str
    cont_names: str = None

    data_path: Path = Path('tbmData/data')
    fn_txt: List[Path] = field(default_factory=list)
    # fn_cycles: Path = Path('tbmData/cycles1')
    fn_cycles: Path = Path('tbmData/cycles-removed-redundent-columns')
    fn_feather: InitVar[str] = None
    fn_np: str = None
    metrics: Callable = None

    debug: bool = True
    gpu_start: int = 2
    num_gpus: int = 1
    sl: int = 30
    postpond: int = 0
    font: str = '/usr/share/fonts/wqy-microhei/wqy-microhei.ttc'
                # '/System/Library/Fonts/PingFang.ttc'
    mulr: int = 7
    valid_ratio: float = 0.2
    is_problem1: bool = True
    extra_var: List[str] = field(default_factory=list)
    dep_var: List[str] = field(default_factory=list)

    load_data: InitVar[bool] = True


    def __post_init__(self, fn_feather, load_data):
        # configure gpu
        torch.cuda.set_device(self.gpu_start)
        device_ids = list(range(self.gpu_start, self.gpu_start+self.num_gpus))
        import os
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=','.join([str(o) for o in
            device_ids])  # specify which GPU(s) to be used

        if fn_feather is None: 
            fn_feather = Path(f'tbmData/feather/{self.exp_name}.feather')
        else:
            fn_feather = Path(fn_feather)
        fn_feather.parent.mkdir(exist_ok=True)
        self.fn_np = str(fn_feather.parent / (fn_feather.stem + '.npz'))
        self.fn_feather = str(fn_feather)
        # super hacky
        if not isinstance(self.fn_cycles, Tuple):
            self.fn_cycles = sorted(Path(self.fn_cycles).glob('cycle*'))
        else:
            self.fn_cycles = self.fn_cycles[1]

        self.num_cycles = 300 if self.debug else len(self.fn_cycles)
        if self.debug: self.fn_cycles = self.fn_cycles[:self.num_cycles]

        # Fill with default dep_vars
        if len(self.dep_var) == 0:
            self.dep_var = dep_var2

        # Load data
        if load_data:
            npz = np.load(self.fn_np)
            self.idx, self.stat_x, self.stat_y, self.stat_extra_x = npz['idx'], npz['stat_x'], npz['stat_y'], npz['stat_extra_x']
            self.metrics = MAPD(self.stat_y)
            self.cyc_cont = pd.read_feather(self.fn_feather)
            self.n_cont = (self.cyc_cont.shape[1] - len(self.dep_var)) // self.sl

        if len(self.fn_txt) == 0: 
            self.fn_txt = sorted(self.data_path.glob('*.txt'))
        
        train_ratio = 1 - self.valid_ratio
        self.train_idx = np.arange(int(self.num_cycles * self.valid_ratio), self.num_cycles)
        self.valid_idx = np.arange(int(self.num_cycles * self.valid_ratio))
        self.train_idx_tile = (self.train_idx[:, None] + np.arange(self.mulr) * self.num_cycles).flatten()
        self.valid_idx_tile = (self.valid_idx[:, None] + np.arange(self.mulr) * self.num_cycles).flatten()  # take from all tiles

        self.bs = int(self.num_cycles * train_ratio)
        
    @classmethod
    def all_columns(cls, exclude_columns=[], dep_var=dep_var1):
        columns = list(set(full_columns[2:]) - set(exclude_columns) - set(dep_var))
        return columns

