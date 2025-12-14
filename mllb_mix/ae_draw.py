from matplotlib import markers
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# ------------------------------ 数据（用数组/列表给出） ------------------------------
#-------------------------------固定的数据----------------------------------------
batch = np.array([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
dGPU_4090     = np.array([ 16, 16, 16, 16, 16, 17, 18, 22, 33, 45,77,133,227,414])

#-------------------------------从kernel.log中提取----------------------------------------
def parse_kernel_log(log_file='kernel.log'):
    """从kernel.log文件中提取数据"""
    # 初始化字典存储数据
    cpu_data = {}
    dgpu_data = {}
    apu_pl_data = {}
    apu_pk_data = {}
    
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, log_file)
    
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 2:
                continue
            
            key = parts[0]
            
            # 解析 MLLB_CPU_batch_X
            if key.startswith('MLLB_CPU_batch_'):
                batch_size = int(key.split('_')[-1])
                cpu_data[batch_size] = int(parts[1])
            
            # 解析 MLLB_dGPU_batch_X
            elif key.startswith('MLLB_dGPU_batch_'):
                batch_size = int(key.split('_')[-1])
                dgpu_data[batch_size] = int(parts[1])
            
            # 解析 MLLB_APU_PL_batch_X
            elif key.startswith('MLLB_APU_PL_batch_'):
                batch_size = int(key.split('_')[-1])
                apu_pl_data[batch_size] = int(parts[1])
            
            # 解析 MLLB_APU_PK_batch_X
            elif key.startswith('MLLB_APU_PK_batch_'):
                batch_size = int(key.split('_')[-1])
                apu_pk_data[batch_size] = int(parts[1])
    
    return cpu_data, dgpu_data, apu_pl_data, apu_pk_data

# 从log文件提取数据
cpu_data, dgpu_data, apu_pl_data, apu_pk_data = parse_kernel_log()

# 根据batch数组提取对应的值
MLLB_CPU_batch = np.array([cpu_data.get(b, 0) for b in batch])
MLLB_dGPU = np.array([dgpu_data.get(b, 0) for b in batch])
MLLB_APU_PL = np.array([apu_pl_data.get(b, 0) for b in batch])

# MLLB_APU_PK: 从log提取，缺失的用999补全
MLLB_APU_PK_list = []
for b in batch:
    if b in apu_pk_data:
        MLLB_APU_PK_list.append(apu_pk_data[b])
    else:
        MLLB_APU_PK_list.append(999)
MLLB_APU_PK = np.array(MLLB_APU_PK_list)
#-------------------------------------------------------------------------------
apu_optimal = np.minimum(MLLB_APU_PL, MLLB_APU_PK)

# set crossover point manually to get best performance,this can get done by onetime profiling.
# crossover point might different for different hardware,.
# crossover_point = 7
# apu_optimal = np.zeros_like(MLLB_APU_PL)
# apu_optimal[:crossover_point] = MLLB_APU_PK[:crossover_point]
# apu_optimal[crossover_point:] = MLLB_APU_PL[crossover_point:]

# ------------------------------ 统一字体大小 ------------------------------
plt.rcParams.update({
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 12,
})

# ------------------------------ 绘图 ------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(batch,  MLLB_CPU_batch, linestyle='-', label='CPU-SSE', color='#1F7DB7')
#ax.plot(batch, MLLB_APU_PL,   marker='s', linestyle='-', label='Per‑Launch Kernel',color='red')
#ax.plot(batch, apu_persistent_kernel, marker='x',       linestyle='-', label='Persistent Kernel',color='red')          # 虚线无点
ax.plot(batch, MLLB_dGPU, marker='s',       linestyle='-', label='dGPU-LAKE-L',color='green')          # 虚线无点
ax.plot(batch, dGPU_4090, marker='x',       linestyle='--', label='dGPU-LAKE-D',color='#34431A')          # 虚线无点
ax.plot(batch, apu_optimal, marker='o',       linestyle='-', label='iGPU-LAIKA',color='red')          # 虚线无点
#ax.plot(batch, gpu_total,   marker='x', linestyle='-', label='GPU ‑ Total')        # 实线有点


# 优势区间：APU 1‑512，GPU 1024‑4096
ax.axvspan(0, 65536, color='pink', alpha=0.5, label='iGPU sweet spot')
# # 标注示例（可按需调整）
# ax.annotate('Persistent Kernel faster\n(~2.3× vs Per‑Launch Kernel)',
#             xy=(12, 26), xytext=(20, 80),
#             arrowprops=dict(arrowstyle='->', lw=0.7),
#             fontsize=14)

# 添加最左侧点的数值显示
ax.annotate(f'{apu_optimal[0]}', xy=(batch[0], apu_optimal[0]), 
            xytext=(batch[0]-2.2, apu_optimal[0]-0.5), fontsize=12,color='red'
            )
ax.annotate(f'{MLLB_dGPU[0]}', xy=(batch[0], MLLB_dGPU[0]), 
            xytext=(batch[0]-2.8, MLLB_dGPU[0]-2), fontsize=12,color='green'
            )
ax.annotate(f'{dGPU_4090[0]}', xy=(batch[0], dGPU_4090[0]), 
            xytext=(batch[0]-2.8, dGPU_4090[0]-1.5), fontsize=12,color='#34431A'
            )
ax.annotate(f'{ MLLB_CPU_batch[0]}', xy=(batch[0],  MLLB_CPU_batch[0]), 
            xytext=(batch[0]-2.2,  MLLB_CPU_batch[0]-0.2), fontsize=12,color='#1F7DB7'
            )

# 添加最右侧点的数值显示
ax.annotate(f'{apu_optimal[-1]}', xy=(batch[-1], apu_optimal[-1]), xytext=(batch[-1]-10000, apu_optimal[-1]+15), fontsize=12,color='red'
            )
ax.annotate(f'{MLLB_dGPU[-1]}', xy=(batch[-1], MLLB_dGPU[-1]), 
xytext=(batch[-1]-15000, MLLB_dGPU[-1]+100), fontsize=12,color='green'
            )
ax.annotate(f'{dGPU_4090[-1]}', xy=(batch[-1], dGPU_4090[-1]), 
xytext=(batch[-1]-15000, MLLB_dGPU[-1]-180), fontsize=12,color='#34431A'
            )            

# 轴与标题
ax.set_xscale('log', base=2)
ax.set_yscale('log', base=10)
# 获取当前时间并格式化
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+8')
ax.set_xlabel(f'Generated at: {current_time}', fontsize=14)
ax.set_ylabel('Time (us)', fontsize=14)
ax.set_ylim(0, 1000)

ax.grid(True, which='both', linestyle=':')
ax.legend(loc='upper left', fontsize=12)

plt.tight_layout()
plt.savefig('./result.pdf', format='pdf', bbox_inches='tight')
plt.close()
