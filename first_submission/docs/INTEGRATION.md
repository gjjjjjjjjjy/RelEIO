# RelEIO 运行手册

## 复算现有 CSV

在仓库根目录执行以下命令。

~~~bash
python tools/final_recompute.py
python tools/make_paper_tables.py
python tools/make_crosspipeline_table.py
python tools/make_final_figures.py
~~~

前三个命令仅使用标准库，绘图需要 numpy 和 matplotlib。输入默认来自仓库 records/，
输出默认落在 reports/。从其他工作目录调用脚本绝对路径也得到相同输入定位。
所有命令接受 --records 和 --out。字段缺失、非法状态、重复运行标识和缺失数据文件会报错。

主要输出：

| 文件 | 内容 |
|---|---|
| reports/statistics.json | 统计定义、总数、Q1/Q2/Q3、阈值和匹配比较 |
| reports/conditions.csv | 每个条件的计数、Wilson 区间及条件误差 |
| reports/success_curves.csv | 每个条件“完成且误差低于阈值”的计数，分母为全部运行 |
| reports/matched_systems.csv | 每对系统相同场景/保留率下的数值崩溃比较 |
| reports/tables/ | 生成的 LaTeX 表格 |
| reports/figures/ | 生成的 PDF 和 SVG 图 |

现有的三个轴分别有独立清洁运行批次，复算时保持这种分组。五个种子值一致不代表不同轴的清洁运行是同一次实验。

## 统计口径

- Q1 计算各部分失败条件的达标比例，再对这些条件取算术平均。一个条件内须同时观察到失败和达标。
- Q2 对清洁条件下的场景误差排序进行比较，报告各对种子之间排序反转比例的中位数。相等的比较单独计数。
- Q3 将单次完成运行的误差差值与同一场景两组完成运行的中位数差值比较。脚本逐一报告清洁条件与各扰动档位的结果。两种差值任一为零时归入相等组。多个比较共用同一次运行，结果是描述性统计。
- 每个条件的失败区间采用 Wilson 方法。跨系统比较使用双方共同场景及保留率下的数值崩溃次数。

## 本地测试

CPU 算子测试使用的包版本为 NumPy 2.0.2、h5py 3.14.0、PyTorch 2.8.0、Numba 0.60.0。

~~~bash
python -m unittest discover -s tests -v
~~~

测试覆盖统计相等值、条件误差、失败分母、错误记录、空窗口、零时间跨度、
坐标重映射、显式时间单位和整个进程矩阵的失败保留。测试使用的临时目录自动删除。

## GPU 矩阵运行入口

scripts/run_matrix.py 负责调度真实适配器，一次条件一次独立进程。
它不会生成模型结果或把非零退出自动标记成数值崩溃。

参数：

- --system：系统名字；
- --scenes：序列名称；
- --seeds：种子或重复编号；
- --axes：keep、time、patch；
- --command：真实适配器的 argv 模板，使用绝对路径；
- --out：本次新建的日志目录。

模板支持 {scene}、{seed}、{keep}、{tq_us}、{patch} 和 {output}。
必须显式使用 scene、seed 以及所选轴的参数。
该接口通过参数数组启动进程，不执行 shell 字符串。
输出目录已存在时直接报错，避免覆盖旧运行。

三条轴默认分别执行 keep=1/.5/.3/.1、tq_us=0/1000/5000/10000、
patch=96/64/32/16；其余参数分别固定为 1、0、96。
每个轴分别执行其清洁条件。

每次进程保存 command.json、stdout.log、stderr.log；
整批保存 execution.csv，包括开始时间、wall_seconds 和 exit_code。
出现非零退出时记录该次结果并继续其他独立运行；结束后整体退出码为 1。
模型、数据、CUDA 或适配器错误必须查日志区分，不能全部归为算法数值失败。
显存、实际输出覆盖率、MPE/ATE 和异常发生阶段由真实适配器记录。

四个 shell 入口使用同一 runner。它们要求 PYTHON、RELEIO_COMMAND_TEMPLATE、
RELEIO_RUN_DIR、RELEIO_SEEDS，位置参数为场景列表。
DEIO 遍历三轴，其他三个入口遍历事件密度轴。
原来的 eval.py 和 run_vo.py 自定义入口没有随仓库提供，因此现在明确要求真实命令，
不把示例路径当作已验证适配器。

## 外部模型适配

| 系统 | 运行所需材料 | 检查内容 |
|---|---|---|
| DEIO | 原 checkout、实际入口、DEVO.pth、数据加载器改动、环境版本 | 三轴配置真正传入；原始事件与 voxel 分别核对 |
| DEVO 消融 | 原 DEIO 关闭 IMU 的配置和实际参数 | 与独立 DEVO 实现区分；明确共用前端与权重 |
| RAMP-VO | 原 checkout、RPG-DAVIS 转换、配置及权重 | scene 确实选择数据；seed 确实由入口使用；帧通道保持原条件 |
| E2VID+TartanVO | 两套环境、重建入口、定制 VO 入口、相机内参与关联逻辑 | 每个重复执行“转换→重建→VO”全过程；使用独立输出目录 |

原发布说明提到的 CUDA 扩展 API 修改、PyTorch 反序列化兼容性、TartanVO/CuPy 兼容性，
都必须对照找回的实际版本重新核实。其文字描述不能代替可执行适配器与环境记录。

## 输入算子

- voxel 默认显式使用 CPU；RELEIO_VOXEL_DEVICE=cuda 显式选择 GPU。
  指定的设备不可用时直接抛错。CUDA 累加的重复性由实际环境验证。
- 空窗口定义为零张量；窗口内时间戳相等时，事件归入首个时间 bin，保留符号累积。
- 重映射使用整数传感器像素索引；非法坐标、非有限值和逆序时间戳显式报错。
- h5_to_e2vid_txt.py 要求 --events-group、--time-unit、--time-origin。
  单位由原始数据定义给出，包含 s/ms/us/ns；不根据跨度猜测。
- keep<1 时必须给 --window-us。筛选保留原时间精度，整数化仅用于窗口种子。
  zero 原点模式需要同步处理图像、IMU 和 ground truth 时间原点。
- window_checksum 保留为历史计数/求和摘要；相同摘要不能证明输入字节一致。
  不新增 hash 机制，诊断时直接核对原数组或明确回放相同中间张量。
