from matplotlib import markers
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from datetime import datetime

# ------------------------------ 数据（用数组/列表给出） ------------------------------
def parse_size(size_str):
    """将大小字符串转换为KB数值"""
    size_str = size_str.strip()
    if size_str.endswith('K'):
        return int(size_str[:-1])
    elif size_str.endswith('M'):
        return int(size_str[:-1]) * 1024
    else:
        return int(size_str)

def load_data_from_kernel_log(log_file='kernel.log'):
    """从 kernel.log 文件加载数据"""
    log_path = os.path.join(os.path.dirname(__file__), log_file)
    
    data_dict = {}
    batch = None
    
    with open(log_path, 'r') as f:
        reader = csv.reader(f)
        # 读取第一行（列标题）
        header = next(reader)
        # 解析 batch 大小
        batch_sizes = [parse_size(size) for size in header[1:]]  # 跳过第一个空列
        batch = np.array(batch_sizes)
        
        # 读取数据行
        for row in reader:
            if not row or not row[0]:
                continue
            key = row[0].strip()
            values = [float(val.strip()) for val in row[1:]]
            data_dict[key] = np.array(values)
    
    return batch, data_dict

# 加载数据
batch, data = load_data_from_kernel_log('kernel.log')


aes_ni_rd = data.get('AESNI_rd', np.array([]))
aes_ni_wt = data.get('AESNI_wt', np.array([]))
aes_ni_dgpu_rd = data.get('LAKE75aesni_rd', np.array([]))  
aes_ni_dgpu_wt = data.get('LAKE75aesni_wt', np.array([]))  
aes_ni_igpu_rd = data.get('LAIKA75aesni_rd', np.array([]))  
aes_ni_igpu_wt = data.get('LAIKA75aesni_wt', np.array([]))  

# ------------------------------ 统一字体大小 ------------------------------
plt.rcParams.update({
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 14,
})

# ------------------------------ 绘图 ------------------------------
fig, ax = plt.subplots(figsize=(7, 4))  # 增加图形宽度和高度
# ax.plot(batch, cpu_rd, marker='x',       linestyle='-', label='CPU Read',color='#1F7DB7')
# ax.plot(batch, cpu_wt, marker='x',       linestyle='--', label='CPU Write',color='#1F7DB7')
# ax.plot(batch, dgpu_rd,   marker='s', linestyle='-', label='dGPU Read',color='green')
# ax.plot(batch, dgpu_wt,   marker='s', linestyle='--', label='dGPU Write',color='green')
# ax.plot(batch, igpu_rd, marker='o',       linestyle='-', label='iGPU Read',color='red')
# ax.plot(batch, igpu_wt, marker='o',       linestyle='--', label='iGPU Write',color='red')

ax.plot(batch, aes_ni_rd, marker='x',       linestyle='-', label='AES-NI Read',color='#1F7DB7')
ax.plot(batch, aes_ni_wt, marker='x',       linestyle='--', label='AES-NI Write',color='#1F7DB7')
ax.plot(batch, aes_ni_dgpu_rd, marker='s',       linestyle='-', label='AES-NI+dGPU Read',color='green')
ax.plot(batch, aes_ni_dgpu_wt, marker='s',       linestyle='--', label='AES-NI+dGPU Write',color='green')
ax.plot(batch, aes_ni_igpu_rd, marker='o',       linestyle='-', label='AES-NI+iGPU Read',color='red')
ax.plot(batch, aes_ni_igpu_wt, marker='o',       linestyle='--', label='AES-NI+iGPU Write',color='red')



#ax.plot(batch, gpu_total,   marker='x', linestyle='-', label='GPU ‑ Total')        # 实线有点

# 优势区间：APU 1‑512，GPU 1024‑4096
# ax.axvspan(8, 256,   color='lightgreen', alpha=0.5, label='Per‑Launch Kernel Advantage')
# ax.axvspan(512, 4096, color='lightblue',  alpha=0.5, label='Persistent Kernel Advantage')

# 标注示例（可按需调整）
# ax.annotate('',
#             xy=(32, 10), xytext=(20, 80),
#             arrowprops=dict(arrowstyle='->', lw=0.7),
#             fontsize=14)

# # 添加最左侧点的数值显示
# ax.annotate(f'{apu_normal[0]}', xy=(batch[0], apu_normal[0]), 
#             xytext=(batch[0]-2, apu_normal[0]+5), fontsize=14,
#             )
# ax.annotate(f'{apu_persistent_kernel[0]}', xy=(batch[0], apu_persistent_kernel[0]), 
#             xytext=(batch[0]-2, apu_persistent_kernel[0]-4), fontsize=14,
#             )

# # 添加最右侧点的数值显示
# ax.annotate(f'{apu_normal[-1]}', xy=(batch[-1], apu_normal[-1]), xytext=(batch[-1]-200, apu_normal[-1]+8), fontsize=14,
#             )
# ax.annotate(f'{apu_persistent_kernel[-1]}', xy=(batch[-1], apu_persistent_kernel[-1]), 
# xytext=(batch[-1]-600, apu_persistent_kernel[-1]-25), fontsize=14,
#             )

# 轴与标题
ax.set_xscale('log', base=2)
ax.set_xlabel('Block Size', fontsize=14)
ax.set_ylabel('Throughput (MB/s)', fontsize=14)
ax.set_ylim(0, 2500)

# 设置X轴显示所有刻度值
ax.set_xticks(batch)
ax.set_xticklabels(batch)
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+8')
ax.set_xlabel(f'Generated at: {current_time}', fontsize=14)
# 自定义x轴标签格式
def format_batch_size(x, pos):
    if x < 1000:
        return f'{int(x)}K'
    else:
        return f'{int(x/1024)}M'

# 设置x轴标签格式
ax.xaxis.set_major_formatter(plt.FuncFormatter(format_batch_size))

# 设置x轴标签旋转角度，使其倾斜显示，并居中对齐
plt.setp(ax.get_xticklabels(), rotation=45, ha='center')

ax.grid(True, which='both', linestyle=':')
ax.legend(loc='upper left', fontsize=12)

# 调整布局，确保x轴标签不被裁剪
plt.tight_layout()
plt.savefig('./result_mix.pdf', format='pdf', bbox_inches='tight')
plt.close()
