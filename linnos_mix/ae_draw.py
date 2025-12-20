#from turtle import color
from matplotlib import markers
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
#-------------------------------Fixed data----------------------------------------
dGPU_4090_layer_0  = np.array([21,21,21,21, 22, 24, 27,  30, 47 ,  71,  133])
dGPU_4090_layer_1  = np.array([47,47,47,48, 50, 52, 61,  89, 124 ,  251,  733  ])
dGPU_4090_layer_2  = np.array([82,82,82,83, 83,85, 96,  146, 207,  427,  1348  ])
batch =      np.array([1 , 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])

# ------------------------------ Read data from kernel.log ------------------------------
def parse_kernel_log(log_file='kernel.log'):
    """Parse data from kernel.log file"""
    data = {
        0: {'APU_PK': {}, 'APU_PL': {}, 'CPU': {}, 'dGPU': {}},
        1: {'APU_PK': {}, 'APU_PL': {}, 'CPU': {}, 'dGPU': {}},
        2: {'APU_PK': {}, 'APU_PL': {}, 'CPU': {}, 'dGPU': {}}
    }
    
    if not os.path.exists(log_file):
        print(f"Warning: file {log_file} does not exist")
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
            
            # Parse key: linnos+{layer}_{type}_batch_{batch_size}
            if 'linnos+' in key and '_batch_' in key:
                key_parts = key.split('_')
                if len(key_parts) >= 4:
                    layer_str = key_parts[0].replace('linnos+', '')
                    try:
                        layer = int(layer_str)
                        if layer not in [0, 1, 2]:
                            continue
                        
                        # Determine type
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
                        
                        # Extract batch_size
                        batch_idx = key_parts.index('batch')
                        if batch_idx + 1 < len(key_parts):
                            batch_size = int(key_parts[batch_idx + 1])
                            data[layer][data_type][batch_size] = time_value
                    except (ValueError, IndexError):
                        continue
    
    return data

def get_data_array(data_dict, batch_sizes):
    """Extract data from data_dict according to batch_sizes order, fill missing values with 999999"""
    result = []
    for bs in batch_sizes:
        if bs in data_dict:
            result.append(data_dict[bs])
        else:
            result.append(999999)
    return np.array(result)

def find_crossover_point(linnos_dGPU, apu_optimal, batch, threshold=1):
    """
    Find the first crossover point position
    
    Parameters:
        linnos_dGPU: dGPU performance array
        apu_optimal: APU optimal performance array
        batch: batch size array
        threshold: tolerance threshold, if the difference is within threshold, still consider apu_optimal as better
    
    Returns:
        crossover_batch: batch value corresponding to the crossover point (aligned to values in batch array), returns None if no crossover point
    """
    # Determine if apu_optimal is better at each point (considering threshold)
    apu_better = apu_optimal <= (linnos_dGPU + threshold)
    
    # Find the first position where it changes from apu_optimal being better to linnos_dGPU being better
    crossover_idx = None
    for i in range(len(apu_better) - 1):
        if apu_better[i] and not apu_better[i + 1]:
            crossover_idx = i
            break
    
    # If crossover point found, return corresponding batch value
    if crossover_idx is not None:
        return batch[crossover_idx]
    
    # If no crossover point found, check if all are apu_optimal better
    if np.all(apu_better):
        return None  # All are apu_optimal better, no crossover point (need to draw entire range)
    
    # If all are linnos_dGPU better, return None (don't draw advantage range)
    if not np.any(apu_better):
        return None
    
    return None

def plot_layer(layer, data_dict, output_file):
    """Plot and save chart for specified layer"""
    # Extract data
    linnos_cpu = get_data_array(data_dict[layer]['CPU'], batch)
    linnos_apu_pl = get_data_array(data_dict[layer]['APU_PL'], batch)
    linnos_dGPU = get_data_array(data_dict[layer]['dGPU'], batch)
    
    # linnos_apu_pk only has data for layer 0, others filled with 999999
    if layer == 0:
        linnos_apu_pk = get_data_array(data_dict[layer]['APU_PK'], batch)
    else:
        linnos_apu_pk = np.array([999999] * len(batch))
    
    # Select corresponding dGPU_4090 data based on layer
    if layer == 0:
        dGPU_4090 = dGPU_4090_layer_0
    elif layer == 1:
        dGPU_4090 = dGPU_4090_layer_1
    else:  # layer == 2
        dGPU_4090 = dGPU_4090_layer_2
    
    apu_optimal = np.minimum(linnos_apu_pl, linnos_apu_pk)
    
    # Unify font size
    plt.rcParams.update({
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 14,
    })
    
    # Plot
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(batch, linnos_cpu, linestyle='-', label='CPU-SSE', color='#1F7DB7')
    ax.plot(batch, linnos_dGPU, marker='s', linestyle='-', label='dGPU-LAKE', color='green')
    ax.plot(batch, dGPU_4090, marker='x', linestyle='--', label='dGPU-LAKE-D', color='#34431A')
    ax.plot(batch, apu_optimal, marker='o', linestyle='-', label='iGPU-LAIKA', color='red')
    
    # Add value display for leftmost point
    ax.annotate(f'{apu_optimal[0]}', xy=(batch[0], apu_optimal[0]), 
                xytext=(batch[0]-0.25, apu_optimal[0]), fontsize=12, color='red')
    ax.annotate(f'{linnos_dGPU[0]}', xy=(batch[0], linnos_dGPU[0]), 
                xytext=(batch[0]-0.25, linnos_dGPU[0]), fontsize=12, color='green')
    ax.annotate(f'{dGPU_4090[0]}', xy=(batch[0], dGPU_4090[0]), 
                xytext=(batch[0]-0.25, dGPU_4090[0]), fontsize=12, color='#34431A')
    ax.annotate(f'{linnos_cpu[0]}', xy=(batch[0], linnos_cpu[0]), 
                xytext=(batch[0]-0.25, linnos_cpu[0]), fontsize=12, color='#1F7DB7')
    
    # Add value display for rightmost point
    ax.annotate(f'{apu_optimal[-1]}', xy=(batch[-1], apu_optimal[-1]), 
                xytext=(batch[-1]+100, apu_optimal[-1]), fontsize=12)
    ax.annotate(f'{dGPU_4090[-1]}', xy=(batch[-1], dGPU_4090[-1]), 
                xytext=(batch[-1]+100, dGPU_4090[-1]), fontsize=12)
    ax.annotate(f'{linnos_dGPU[-1]}', xy=(batch[-1], linnos_dGPU[-1]), 
                xytext=(batch[-1]+100, linnos_dGPU[-1]), fontsize=12)
    ax.annotate(f'{linnos_cpu[-1]}', xy=(batch[-1], linnos_cpu[-1]), 
                xytext=(batch[-1]+100, linnos_cpu[-1]), fontsize=12)
    
    # Automatically find advantage range
    crossover_batch = find_crossover_point(linnos_dGPU, apu_optimal, batch, threshold=2)
    apu_better = apu_optimal <= (linnos_dGPU + 2)
    
    if crossover_batch is not None:
        # Found crossover point, draw advantage range from 0 to crossover point
        ax.axvspan(0, crossover_batch, color='pink', alpha=0.5)
        #print(f"Layer {layer}: Advantage range is batch 0 to {crossover_batch}")
    elif np.all(apu_better):
        # All are apu_optimal better, draw entire range
        ax.axvspan(0, batch[-1], color='pink', alpha=0.5)
        #print(f"Layer {layer}: Advantage range is entire range (batch {batch[0]} to {batch[-1]})")
    else:
        # All are linnos_dGPU better, don't draw advantage range
        print(f"Layer {layer}: No advantage range (linnos_dGPU is always better)")
    
    # Axes and title
    ax.set_xscale('log', base=2)
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+8')
    ax.set_xlabel(f'Generated at: {current_time}', fontsize=14)
    ax.set_yscale('log', base=10)
    ax.set_ylabel('Time (us)', fontsize=14)
    # For log scale, set a small positive number as lower limit
    ax.set_ylim(0, 1200)
    
    ax.grid(True, which='both', linestyle=':')
    
    plt.tight_layout()
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

# ------------------------------ Main program ------------------------------
if __name__ == '__main__':
    # Read data
    data = parse_kernel_log('kernel.log')
    
    # Generate charts for each layer
    for layer in [0, 1, 2]:
        output_file = f'layer_plus_{layer}.pdf'
        plot_layer(layer, data, output_file)
    
    print("All charts generated successfully!")
