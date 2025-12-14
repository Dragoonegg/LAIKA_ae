# Part of LAKE: Towards a Machine Learning-Assisted Kernel with LAKE
# Original work:
# Copyright (C) 2022–2024 Henrique Fingler
# Copyright (C) 2022–2024 Isha Tarte
# Modifications and adaptations for LAIKA:
# Copyright (C) 2024-2025 Haoming Zhuo
# This file is adapted from the original LAKE kernel module.
# Major changes include:
# - Integration with LAIKA framework
# - Hybrid execution support
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

cpu_rd = data.get('CPU_rd', np.array([]))
cpu_wt = data.get('CPU_wt', np.array([]))
dgpu_rd = data.get('LAKE_rd', np.array([]))  # dgpu对应LAKE
dgpu_wt = data.get('LAKE_wt', np.array([]))  # dgpu对应LAKE
igpu_rd = data.get('LAIKA_rd', np.array([]))  # igpu对应LAIKA
igpu_wt = data.get('LAIKA_wt', np.array([]))  # igpu对应LAIKA


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
ax.plot(batch, cpu_rd, marker='x',       linestyle='-', label='CPU Read',color='#1F7DB7')
ax.plot(batch, cpu_wt, marker='x',       linestyle='--', label='CPU Write',color='#1F7DB7')
ax.plot(batch, dgpu_rd,   marker='s', linestyle='-', label='dGPU Read',color='green')
ax.plot(batch, dgpu_wt,   marker='s', linestyle='--', label='dGPU Write',color='green')
ax.plot(batch, igpu_rd, marker='o',       linestyle='-', label='iGPU Read',color='red')
ax.plot(batch, igpu_wt, marker='o',       linestyle='--', label='iGPU Write',color='red')

# ax.plot(batch, aes_ni_rd, marker='x',       linestyle='-', label='AES-NI Read',color='black')
# ax.plot(batch, aes_ni_wt, marker='x',       linestyle='--', label='AES-NI Write',color='black')


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
ax.set_ylim(0, 2000)

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
plt.savefig('./result_raw.pdf', format='pdf', bbox_inches='tight')
plt.close()
