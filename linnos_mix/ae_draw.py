#from turtle import color
from matplotlib import markers
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
#-------------------------------固定的数据----------------------------------------
dGPU_4090_layer_0  = np.array([21,21,21,21, 22, 24, 27,  30, 47 ,  71,  133])
dGPU_4090_layer_1  = np.array([47,47,47,48, 50, 52, 61,  89, 124 ,  251,  733  ])
dGPU_4090_layer_2  = np.array([82,82,82,83, 83,85, 96,  146, 207,  427,  1348  ])
batch =      np.array([1 , 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])

# ------------------------------ 从 kernel.log 读取数据 ------------------------------
def parse_kernel_log(log_file='kernel.log'):
    """从 kernel.log 文件中解析数据"""
    data = {
        0: {'APU_PK': {}, 'APU_PL': {}, 'CPU': {}, 'dGPU': {}},
        1: {'APU_PK': {}, 'APU_PL': {}, 'CPU': {}, 'dGPU': {}},
        2: {'APU_PK': {}, 'APU_PL': {}, 'CPU': {}, 'dGPU': {}}
    }
    
    if not os.path.exists(log_file):
        print(f"警告: 文件 {log_file} 不存在")
        return data
    
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) != 2:
                continue
            
            key, value = parts
            try:
                time_value = int(value)
            except ValueError:
                continue
            
            # 解析 key: linnos+{layer}_{type}_batch_{batch_size}
            if 'linnos+' in key and '_batch_' in key:
                key_parts = key.split('_')
                if len(key_parts) >= 4:
                    layer_str = key_parts[0].replace('linnos+', '')
                    try:
                        layer = int(layer_str)
                        if layer not in [0, 1, 2]:
                            continue
                        
                        # 确定类型
                        if 'APU_PK' in key:
                            data_type = 'APU_PK'
                        elif 'APU_PL' in key:
                            data_type = 'APU_PL'
                        elif 'CPU' in key:
                            data_type = 'CPU'
                        elif 'dGPU' in key:
                            data_type = 'dGPU'
                        else:
                            continue
                        
                        # 提取 batch_size
                        batch_idx = key_parts.index('batch')
                        if batch_idx + 1 < len(key_parts):
                            batch_size = int(key_parts[batch_idx + 1])
                            data[layer][data_type][batch_size] = time_value
                    except (ValueError, IndexError):
                        continue
    
    return data

def get_data_array(data_dict, batch_sizes):
    """根据 batch_sizes 顺序从 data_dict 中提取数据，缺失的用 999999 填充"""
    result = []
    for bs in batch_sizes:
        if bs in data_dict:
            result.append(data_dict[bs])
        else:
            result.append(999999)
    return np.array(result)

def find_crossover_point(linnos_dGPU, apu_optimal, batch, threshold=1):
    """
    找到第一个交叉点位置
    
    参数:
        linnos_dGPU: dGPU性能数组
        apu_optimal: APU最优性能数组
        batch: batch大小数组
        threshold: 容忍阈值，如果差距在threshold以内，仍然算apu_optimal较优
    
    返回:
        crossover_batch: 交叉点对应的batch值（对齐到batch数组中的值），如果没有交叉点则返回None
    """
    # 判断每个点apu_optimal是否较优（考虑阈值）
    apu_better = apu_optimal <= (linnos_dGPU + threshold)
    
    # 找到第一个从apu_optimal较优变为linnos_dGPU较优的位置
    crossover_idx = None
    for i in range(len(apu_better) - 1):
        if apu_better[i] and not apu_better[i + 1]:
            crossover_idx = i
            break
    
    # 如果找到交叉点，返回对应的batch值
    if crossover_idx is not None:
        return batch[crossover_idx]
    
    # 如果没有找到交叉点，检查是否全部都是apu_optimal较优
    if np.all(apu_better):
        return None  # 全部都是apu_optimal较优，没有交叉点（需要绘制整个区间）
    
    # 如果全部都是linnos_dGPU较优，返回None（不绘制优势区间）
    if not np.any(apu_better):
        return None
    
    return None

def plot_layer(layer, data_dict, output_file):
    """为指定 layer 绘制图表并保存"""
    # 提取数据
    linnos_cpu = get_data_array(data_dict[layer]['CPU'], batch)
    linnos_apu_pl = get_data_array(data_dict[layer]['APU_PL'], batch)
    linnos_dGPU = get_data_array(data_dict[layer]['dGPU'], batch)
    
    # linnos_apu_pk 只有 layer 0 有数据，其他用 999999 填充
    if layer == 0:
        linnos_apu_pk = get_data_array(data_dict[layer]['APU_PK'], batch)
    else:
        linnos_apu_pk = np.array([999999] * len(batch))
    
    # 根据 layer 选择对应的 dGPU_4090 数据
    if layer == 0:
        dGPU_4090 = dGPU_4090_layer_0
    elif layer == 1:
        dGPU_4090 = dGPU_4090_layer_1
    else:  # layer == 2
        dGPU_4090 = dGPU_4090_layer_2
    
    apu_optimal = np.minimum(linnos_apu_pl, linnos_apu_pk)
    
    # 统一字体大小
    plt.rcParams.update({
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 14,
    })
    
    # 绘图
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(batch, linnos_cpu, linestyle='-', label='CPU-SSE', color='#1F7DB7')
    ax.plot(batch, linnos_dGPU, marker='s', linestyle='-', label='dGPU-LAKE', color='green')
    ax.plot(batch, dGPU_4090, marker='x', linestyle='--', label='dGPU-LAKE-D', color='#34431A')
    ax.plot(batch, apu_optimal, marker='o', linestyle='-', label='iGPU-LAIKA', color='red')
    
    # 添加最左侧点的数值显示
    ax.annotate(f'{apu_optimal[0]}', xy=(batch[0], apu_optimal[0]), 
                xytext=(batch[0]-0.25, apu_optimal[0]), fontsize=12, color='red')
    ax.annotate(f'{linnos_dGPU[0]}', xy=(batch[0], linnos_dGPU[0]), 
                xytext=(batch[0]-0.25, linnos_dGPU[0]), fontsize=12, color='green')
    ax.annotate(f'{dGPU_4090[0]}', xy=(batch[0], dGPU_4090[0]), 
                xytext=(batch[0]-0.25, dGPU_4090[0]), fontsize=12, color='#34431A')
    ax.annotate(f'{linnos_cpu[0]}', xy=(batch[0], linnos_cpu[0]), 
                xytext=(batch[0]-0.25, linnos_cpu[0]), fontsize=12, color='#1F7DB7')
    
    # 添加最右侧点的数值显示
    ax.annotate(f'{apu_optimal[-1]}', xy=(batch[-1], apu_optimal[-1]), 
                xytext=(batch[-1]+100, apu_optimal[-1]), fontsize=12)
    ax.annotate(f'{dGPU_4090[-1]}', xy=(batch[-1], dGPU_4090[-1]), 
                xytext=(batch[-1]+100, dGPU_4090[-1]), fontsize=12)
    ax.annotate(f'{linnos_dGPU[-1]}', xy=(batch[-1], linnos_dGPU[-1]), 
                xytext=(batch[-1]+100, linnos_dGPU[-1]), fontsize=12)
    ax.annotate(f'{linnos_cpu[-1]}', xy=(batch[-1], linnos_cpu[-1]), 
                xytext=(batch[-1]+100, linnos_cpu[-1]), fontsize=12)
    
    # 自动寻找优势区间
    crossover_batch = find_crossover_point(linnos_dGPU, apu_optimal, batch, threshold=2)
    apu_better = apu_optimal <= (linnos_dGPU + 2)
    
    if crossover_batch is not None:
        # 找到交叉点，绘制从0到交叉点的优势区间
        ax.axvspan(0, crossover_batch, color='pink', alpha=0.5)
        #print(f"Layer {layer}: 优势区间为 batch 0 到 {crossover_batch}")
    elif np.all(apu_better):
        # 全部都是apu_optimal较优，绘制整个区间
        ax.axvspan(0, batch[-1], color='pink', alpha=0.5)
        #print(f"Layer {layer}: 优势区间为整个范围 (batch {batch[0]} 到 {batch[-1]})")
    else:
        # 全部都是linnos_dGPU较优，不绘制优势区间
        print(f"Layer {layer}: 没有优势区间（linnos_dGPU始终较优）")
    
    # 轴与标题
    ax.set_xscale('log', base=2)
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+8')
    ax.set_xlabel(f'Generated at: {current_time}', fontsize=14)
    ax.set_yscale('log', base=10)
    ax.set_ylabel('Time (us)', fontsize=14)
    # 对于 log scale，设置一个小的正数作为下限
    ax.set_ylim(0, 1200)
    
    ax.grid(True, which='both', linestyle=':')
    
    plt.tight_layout()
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_file}")

# ------------------------------ 主程序 ------------------------------
if __name__ == '__main__':
    # 读取数据
    data = parse_kernel_log('kernel.log')
    
    # 为每个 layer 生成图表
    for layer in [0, 1, 2]:
        output_file = f'layer_plus_{layer}.pdf'
        plot_layer(layer, data, output_file)
    
    print("所有图表已生成完成！")
