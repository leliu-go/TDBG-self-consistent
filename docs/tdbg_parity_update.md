# TDBG 本地更新（2026-10-10）

本次按用户要求保留所有原物理参数，只更新已确认的数值检查、Stoner 搜索和结果流程。
计算仍是完整连续模型层 Hartree，再在固定 Hartree 势上做四味 Stoner；不恢复八带投影。

## 保留的模型与参数

- 原哈密顿量矩阵、基底顺序、所有跃迁和二聚体在位能不变。
- 默认 theta=1.35 度；gamma0/1/3/4=2610/361/283/138 meV，Z=15 meV。
- omega=100 meV、wAA=0.8、wAB=1、a_cc=1.42 Å、层间距 0.335 nm。
- Hartree 独立入口 eps_perp=4；Stoner/Windows 入口 eps_perp=3，保持各自原值。
- D_sign=-1、kBT=0.2 meV、U0=79000、JH=24000 meV Å²不变。
- Windows 原默认 20 workers、N=2、36×36、nu=-6...6/61、D=-0.8...0.8/41、detail=10 不变。
- Python nD 原默认 D=-1.2...1.2/121 保留，不能误认为所有入口原默认相同。

固定命名 AB=B上-A下，BA=A上-B下时，原四层的实际二聚体是 A1-B2、B3-A4，
所以元数据明确标为从上到下 BA-AB。旧模型标识与文件夹前缀 ABBA 保留，不改变任何耦合。
D>0 默认从下到上；z 从上到下，电子势能 U=-e*phi。
无层电荷时 U[l+1]-U[l]=D_sign*1000*d_layer_nm*D/eps_perp（meV）。

## 数值与 Stoner

修正局部稳定性测试只走到可行区间 80% 而可能漏掉端点降能的问题；现在检查真实二次段最小值及端点。
采用三维差分进化的两个独立种子、占据节点精修、共同费米能与可行稳定性检查，再比较能量。
默认每种子 DE 150 代、popsize=12，最多 200 步节点精修；种子为 0/1。
保留原 Stoner 能量泛函、U0/JH 和每味净填充 [-2,2]。旧 random-seed 数输入仅用于读取旧配置，不参与 DE。
诊断保留每个候选的能量、精修、驻点和稳定性状态；DE 不构成数学全局最小值证明。

Hartree 求解、Anderson 混合和零温目标感知的分数占据保持原实现。此前已正确的物理源码不重写。
Stoner 仍不反馈更新 Hartree，也没有加入有限 Q。

## 输出与恢复

- 活性自旋、谷、自旋谷极化用 *_polarization_active_norm，范围 0...1；无活性载流子时为 NaN。
  它们来自全谱填充后的中央导带电子 e_f 和价带空穴 h_f：
  P_a=abs(sum_f q_a,f*(e_f-h_f))/sum_f(e_f+h_f)。远带填充单独记录。
  旧 *_polarization_norm 按绝对净填充归一化，作为旧列保留并明确标注；不能把它当作有界活性极化。
- scf_/stoner_layer_dos_lower_block_fraction 为 (DOS_L3+DOS_L4)/DOS_all。
  能带/等能线颜色为完整本征态在 L3-L4 上的概率，含全部平面波与两亚晶格。
- Hartree nu=0、±4 能隙，Stoner 整数参考填充 -6...6 能隙；无隙为 0，未算或无效为 NaN。
  参考能隙在每点固定势和固定交换位移的能谱上求，不代表不同填充的独立自洽解。
- 主流程详细点只保存压缩能谱及紧凑概率，不保存完整本征矢；仍保留纯 Hartree 和 Stoner 两阶段。
  Origin TXT 含 Long Name/Units/Comments 三行，并标出 mu；自动导出分块着色 PDF/PNG。
  C3 补全仅为明确标记的绘图插值，原始网格数据不变。低于 2×2 的网格只导出原始等能数据，标记插值不可用。
- 每点落盘、实时 map、时间和 ETA、校验和及原子替换保留；中断后恢复未完成点。
- 已收敛 Hartree 可单独重算 Stoner，Hartree 迭代为 0；仍需在保存势上对角化两谷并验残差。
  同目录重算先归档旧结果；正在更新时 Hartree map 保留完整，未重算的 Stoner 点为 NaN。
- 修复失败重算残留旧 Origin 图表，以及续算中进程传输失败掩盖既有有效 Hartree 数据的情况。

主结果仍默认写入 D:/LeLiu/data/tdbg_self_consistent。每次保存 config_used.json、哈密顿量与参数、
源代码指纹及诊断。磁盘预算在启动时按本次网格、维数和详细点数估计；历史备份另占空间。

服务器命令见 ../server_full_hartree_stoner_cli.md；各入口原有物理参数保持原值。

## 验证记录

2026-10-10，相关回归共 337 项通过，用时 886.06 秒；有限 Q 的 susceptibility 测试不在本轮范围内。
检查覆盖原哈密顿量、完整 Hartree、零温分数占据、Stoner 搜索与接受条件、极化和能隙、
四层投影、Origin 导出，以及实际 Windows 启动、同目录重算和中断续算。
实际两点 PowerShell 重算成功，Hartree 迭代次数为 0；两阶段详细图表也已实际生成，并检查代表图的布局。

continuum.py、density.py、electrostatics.py、mixing.py、solver_full.py 和 run_full_scf.py
的 SHA-256 均与修改前一致；原物理参数和 Windows 默认参数已逐项核对。
数值测试环境为 Python 3.11.7、NumPy 1.24.4、SciPy 1.10.1；另外检查了全部 62 个生产 Python 文件的
Python 3.8 语法兼容性，这不等同于已经在服务器的 Python 3.8 环境重跑整个测试。

默认 36×36 网格、2501 个扫描点、10 个详细点的启动保存预算约 1.44 GB。
该数值是文件预算，运行内存和历史归档另计。完整本征矢仍在计算时使用，只是不写入详细点结果。
