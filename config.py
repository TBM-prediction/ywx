from utils.misc import *
from preprocessing import *
from mymodels import *
from databunch import *
from crits import *

# import jtplot submodule from jupyterthemes
# currently installed theme will be used to
# set plot style if no arguments provided
from jupyterthemes import jtplot
jtplot.style()

from typing import Callable


dep_var1 = ['推进速度电位器设定值', '刀盘转速电位器设定值']

full_columns = ['时间戳', '运行时间', '桩号', '前点偏差X', '前点偏差Y',
        '前盾俯仰角', '前盾滚动角', '主液压油箱温度', '液压油箱温度预警设置',
        '液压油箱温度报警设置', '温度误差设置', '左侧护盾压力', '右侧护盾压力',
        '顶护盾压力', '左侧楔形油缸压力', '右侧楔形油缸压力',
        '左侧扭矩油缸伸出压力', '左侧扭矩油缸回收压力', '右侧扭矩油缸伸出压力',
        '右侧扭矩油缸回收压力', '左侧后支撑压力', '右侧后支撑压力', '撑靴压力',
        '左侧护盾位移', '右侧护盾位移', '左侧扭矩油缸位移', '右侧扭矩油缸位移',
        '内循环水罐温度', '内循环水罐液位', '冷水箱温度', '冷水箱液位',
        '热水箱温度', '热水箱液位', '刀盘喷水压力', '污水箱压力检测',
        'TBM外水进水温度', 'TBM主冷却器进水温度', '暖水泵压力', '冷水泵压力',
        '内水泵压力', '变频器1温度', '变频器2温度', '暖水箱自动排水温度设置',
        '二次风机频率设置', '齿轮油温度', 'EP2泵 出口压力检测', 'EP2\
 内密封压力', 'EP2 外密封压力', '齿轮润滑油箱压力液位1',
        '齿轮润滑油箱压力液位2', '泵1润滑压力', '泵2润滑压力', '泵3润滑压力',
        '泵4润滑压力', '齿轮密封压力', '齿轮回油泵出口压力', '主驱动加压压力',
        '润滑泵电机电流', '外密封腔流量', '外密封流量', '齿轮润滑流量1',
        '齿轮润滑流量2', '前部小齿轮轴承润滑流量1', '前部小齿轮轴承润滑流量2',
        '后部小齿轮轴承润滑流量1', '后部小齿轮轴承润滑流量2',
        '齿轮密封外密封压力', '齿轮密封内密封压力', '齿轮油温度预警设置',
        '齿轮油温度报警设置', '主驱动1#电机电流', '主驱动2#电机电流',
        '主驱动3#电机电流', '主驱动4#电机电流', '主驱动5#电机电流',
        '主驱动6#电机电流', '主驱动7#电机电流', '主驱动8#电机电流',
        '主驱动9#电机电流', '主驱动10#电机电流', '主驱动1#电机扭矩',
        '主驱动2#电机扭矩', '主驱动3#电机扭矩', '主驱动4#电机扭矩',
        '主驱动5#电机扭矩', '主驱动6#电机扭矩', '主驱动7#电机扭矩',
        '主驱动8#电机扭矩', '主驱动9#电机扭矩', '主驱动10#电机扭矩',
        '主驱动1#电机输出频率', '主驱动2#电机输出频率', '主驱动3#电机输出频率',
        '主驱动4#电机输出频率', '主驱动5#电机输出频率', '主驱动6#电机输出频率',
        '主驱动7#电机输出频率', '主驱动8#电机输出频率', '主驱动9#电机输出频率',
        '主驱动10#电机输出频率', '主驱动1#电机输出功率',
        '主驱动2#电机输出功率', '主驱动3#电机输出功率', '主驱动4#电机输出功率',
        '主驱动5#电机输出功率', '主驱动6#电机输出功率', '主驱动7#电机输出功率',
        '主驱动8#电机输出功率', '主驱动9#电机输出功率',
        '主驱动10#电机输出功率', '减速机1#温度', '减速机2#温度',
        '减速机3#温度', '减速机4#温度', '减速机5#温度', '减速机6#温度',
        '减速机7#温度', '减速机8#温度', '减速机9#温度', '减速机10#温度',
        '刀盘转速', '刀盘转速电位器设定值', '刀盘刹车压力', '刀盘扭矩',
        '刀盘给定转速显示值', '刀盘速度给定', '刀盘功率', '刀盘运行时间',
        '刀盘运行时间.1', '给定频率', '变频柜回水温度报警值',
        '变频柜回水温度停机值', '减速机温度报警值', '减速机温度停机值',
        '刀盘最低转速设置', '左撑靴小腔压力', '右撑靴小腔压力', '推进压力',
        '贯入度', '总推进力', '推进速度', '推进位移', '推进泵电机电流',
        '撑靴泵电机电流', '换步泵电机电流', '左推进油缸行程检测',
        '右推进油缸行程检测', '左撑靴油缸行程检测', '右撑靴油缸行程检测',
        '推进速度电位器设定值', '推进泵压力', '控制油路2压力检测',
        '辅助系统压力检测', '推进速度给定百分比', '左撑靴俯仰角',
        '左撑靴滚动角', '右撑靴俯仰角', '右撑靴滚动角', '控制泵压力',
        '控制油路1 压力', '换步泵1 压力', '撑靴泵压力', '换步泵2 压力',
        '钢拱架泵压力', '主机皮带机泵压力', '左倾最大滚动角设置',
        '右倾最大滚动角设置', '左撑靴最大滚动角设置', '左撑靴最大俯仰角设置',
        '右撑靴最大滚动角设置', '右撑靴最大俯仰角设置', '撑靴压力设定',
        '推进位移最大允许偏差设置', '贯入度设置', '推进给定速度百分比', '推进速度.1',
        '刀盘CH4气体浓度', '刀盘H2S浓度', '控制室O2浓度', '控制室CO浓度',
        '控制室CO2浓度', '设备桥CH4浓度', '拖车尾部CH4浓度', '左拖拉油缸压力',
        '右拖拉油缸压力', '拖拉油缸最大允许压力设置', '拖拉油缸最小允许压力设置',
        '机械手泵1 电机电流', '机械手泵2 电机电流', '主机皮带机转速', '桥架皮带机转速',
        '转渣皮带机转速', '主机皮带机泵电机电流', '主皮带机转速电位器设定值',
        '桥架皮带机转速电位器设定值', '转渣皮带机转速电位器设定值']

@dataclass
class Context:

    exp_name: str
    cont_names: str

    data_path: Path = Path('tbmData/data')
    fn_txt: List[Path] = field(default_factory=list)
    fn_cycles: Path = Path('tbmData/cycles1')
    fn_feather: InitVar[str] = None
    fn_np: str = None
    metrics: Callable = None

    debug: bool = True
    gpu_start: int = 2
    sl: int = 30
    postpond: int = 0
    font: str = '/usr/share/fonts/wqy-microhei/wqy-microhei.ttc'
                # '/System/Library/Fonts/PingFang.ttc'
    mulr: int = 7
    valid_ratio: float = 0.2
    is_problem1: bool = True

    load_data: InitVar[bool] = True


    def __post_init__(self, fn_feather, load_data):
        if fn_feather is None: 
            fn_feather = Path(f'tbmData/feather/{self.exp_name}.feather')
        else:
            fn_feather = Path(fn_feather)
        fn_feather.parent.mkdir(exist_ok=True)
        self.fn_np = str(fn_feather.parent / (fn_feather.stem + '.npz'))
        self.fn_feather = str(fn_feather)
        self.num_cycles = 300 if self.debug else 3481
        self.fn_cycles = sorted(self.fn_cycles.glob('cycle*'))[:self.num_cycles]

        # load data
        if load_data:
            npz = np.load(self.fn_np)
            self.idx, self.stat_x, self.stat_y = npz['idx'], npz['stat_x'], npz['stat_y']
            self.metrics = MAPD(self.stat_y)
            self.cyc_cont = pd.read_feather(self.fn_feather)

        if len(self.fn_txt) == 0: 
            self.fn_txt = sorted(self.data_path.glob('*.txt'))
        self.n_cont = len(self.cont_names)
        
        train_ratio = 1 - self.valid_ratio
        self.train_idx = np.arange(int(self.num_cycles * self.valid_ratio), self.num_cycles)
        self.valid_idx = np.arange(int(self.num_cycles * self.valid_ratio))
        self.train_idx_tile = (self.train_idx[:, None] + np.arange(self.mulr) * self.num_cycles).flatten()
        self.valid_idx_tile = (self.valid_idx[:, None] + np.arange(self.mulr) * self.num_cycles).flatten()  # take from all tiles

        self.bs = int(self.num_cycles * train_ratio)
        
        if self.is_problem1:
            self.dep_var = dep_var1
        else:
            raise NotImplementedError

    @classmethod
    def all_columns(cls, dep_var=dep_var1):
        if dep_var is not None:
            # TODO: I think the test data won't include 桩号 so consider remove that from training data as well
            all_columns = list(set(full_columns[2:]) - set(dep_var))
        return all_columns

