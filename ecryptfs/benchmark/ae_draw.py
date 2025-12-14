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

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(batch, cpu_rd, marker='x', linestyle='-', label='CPU Read', color='#1F7DB7')
    ax.plot(batch, cpu_wt, marker='x', linestyle='--', label='CPU Write', color='#1F7DB7')
    ax.plot(batch, dgpu_rd, marker='s', linestyle='-', label='dGPU Read', color='green')
    ax.plot(batch, dgpu_wt, marker='s', linestyle='--', label='dGPU Write', color='green')
    ax.plot(batch, igpu_rd, marker='o', linestyle='-', label='iGPU Read', color='red')
    ax.plot(batch, igpu_wt, marker='o', linestyle='--', label='iGPU Write', color='red')

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
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.close()
    print(f'已生成: {output_file}')

# ------------------------------ 绘图函数：mix ------------------------------
def draw_mix(batch, data, output_file='./result_mix.pdf'):
    """绘制混合数据图（AES-NI相关）"""
    aes_ni_rd = data.get('AESNI_rd', np.array([]))
    aes_ni_wt = data.get('AESNI_wt', np.array([]))
    aes_ni_dgpu_rd = data.get('LAKE75aesni_rd', np.array([]))  
    aes_ni_dgpu_wt = data.get('LAKE75aesni_wt', np.array([]))  
    aes_ni_igpu_rd = data.get('LAIKA75aesni_rd', np.array([]))  
    aes_ni_igpu_wt = data.get('LAIKA75aesni_wt', np.array([]))  

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(batch, aes_ni_rd, marker='x', linestyle='-', label='AES-NI Read', color='#1F7DB7')
    ax.plot(batch, aes_ni_wt, marker='x', linestyle='--', label='AES-NI Write', color='#1F7DB7')
    ax.plot(batch, aes_ni_dgpu_rd, marker='s', linestyle='-', label='AES-NI+dGPU Read', color='green')
    ax.plot(batch, aes_ni_dgpu_wt, marker='s', linestyle='--', label='AES-NI+dGPU Write', color='green')
    ax.plot(batch, aes_ni_igpu_rd, marker='o', linestyle='-', label='AES-NI+iGPU Read', color='red')
    ax.plot(batch, aes_ni_igpu_wt, marker='o', linestyle='--', label='AES-NI+iGPU Write', color='red')

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
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.close()
    print(f'已生成: {output_file}')

# ------------------------------ 主程序 ------------------------------
if __name__ == '__main__':
    # 加载数据
    batch, data = load_data_from_kernel_log('kernel.log')
    
    # 生成两张图
    draw_raw(batch, data)
    draw_mix(batch, data)
    
    print('所有图表已生成完成！')

