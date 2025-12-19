# from turtle import color
from matplotlib import markers
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# ------------------------------ Data (provided as arrays/lists) ------------------------------
#-------------------------------Fixed data----------------------------------------
dGPU_4090  = np.array([25,25,25,26, 26, 26, 27,  28, 30 ,  34,  42,   59,  101 ])
batch =      np.array([1 , 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
#-------------------------------Extracted from kernel.log----------------------------------------


def parse_kernel_log(log_file='kernel.log'):
    """Extract data from kernel.log file"""
    # Initialize dictionary to store data
    cpu_data = {}
    dgpu_data = {}
    apu_pl_data = {}
    apu_pk_data = {}
    
    # Get the directory where the current script is located
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
            
            # Parse KML_CPU_batch_X
            if key.startswith('KML_CPU_batch_'):
                batch_size = int(key.split('_')[-1])
                cpu_data[batch_size] = int(parts[1])
            
            # Parse KML_dGPU_batch_X
            elif key.startswith('KML_dGPU_batch_'):
                batch_size = int(key.split('_')[-1])
                dgpu_data[batch_size] = int(parts[1])
            
            # Parse KML_APU_PL_batch_X
            elif key.startswith('KML_APU_PL_batch_'):
                batch_size = int(key.split('_')[-1])
                apu_pl_data[batch_size] = int(parts[1])
            
            # Parse KML_APU_PK_batch_X
            elif key.startswith('KML_APU_PK_batch_'):
                batch_size = int(key.split('_')[-1])
                apu_pk_data[batch_size] = int(parts[1])
    
    return cpu_data, dgpu_data, apu_pl_data, apu_pk_data

# Extract data from log file
cpu_data, dgpu_data, apu_pl_data, apu_pk_data = parse_kernel_log()

# Extract corresponding values based on batch array
KML_CPU_batch = np.array([cpu_data.get(b, 0) for b in batch])
KML_dGPU = np.array([dgpu_data.get(b, 0) for b in batch])
KML_APU_PL = np.array([apu_pl_data.get(b, 0) for b in batch])

# MLLB_APU_PK: Extract from log, fill missing values with 999
KML_APU_PK_list = []
for b in batch:
    if b in apu_pk_data:
        KML_APU_PK_list.append(apu_pk_data[b])
    else:
        KML_APU_PK_list.append(999)
KML_APU_PK = np.array(KML_APU_PK_list)
#-------------------------------------------------------------------------------
apu_optimal = np.minimum(KML_APU_PL, KML_APU_PK)
# set crossover point manually to get best performance,this can get done by onetime profiling.
# crossover point might different for different hardware,.

def find_crossover_point(arr1, arr2, batch_values):

    crossover_idx = None
    
    for i in range(len(arr1) - 1):
        # Check if crossover occurs: from arr1 < arr2 to arr1 >= arr2
        if arr1[i] < arr2[i] and arr1[i+1] >= arr2[i+1]:
            crossover_idx = i + 1
            break
        # Or from arr1 > arr2 to arr1 <= arr2
        elif arr1[i] > arr2[i] and arr1[i+1] <= arr2[i+1]:
            crossover_idx = i + 1
            break
    
    # If no crossover point found, check if arr1 is always less than arr2 or always greater than arr2
    if crossover_idx is None:
        # If arr1 is always less than arr2, return the last batch value
        if np.all(arr1 < arr2):
            return batch_values[-1]
        # If arr1 is always greater than arr2, return the first batch value
        elif np.all(arr1 > arr2):
            return batch_values[0]
        # If completely equal, return the middle value
        else:
            return batch_values[len(batch_values) // 2]
    
    # Map the crossover point to the closest power-of-2 value in the batch array
    crossover_batch = batch_values[crossover_idx]
    
    return crossover_batch

# Automatically find crossover point
crossover_batch = find_crossover_point(apu_optimal, KML_dGPU, batch)


# ------------------------------ Uniform font size ------------------------------
plt.rcParams.update({
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 14,
})

# ------------------------------ Plotting ------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(batch, KML_CPU_batch, linestyle='-', label='CPU-SSE',color='#1F7DB7')
#ax.plot(batch, apu_normal,   marker='s', linestyle='-', label='Per‑Launch Kernel',color='red')
#ax.plot(batch, apu_persistent_kernel, marker='x',       linestyle='-', label='Persistent Kernel',color='red')          # Dashed line without points
ax.plot(batch, KML_dGPU, marker='s',       linestyle='-', label='dGPU-LAKE-L',color='green')
ax.plot(batch, dGPU_4090, marker='x',       linestyle='--', label='dGPU-LAKE-D',color='#34431A')          # Dashed line without points
#ax.plot(batch, dGPU_4090, marker='s',       linestyle='--', label='dGPU-LAKE-D',color='green')          # Dashed line without points
ax.plot(batch, apu_optimal, marker='o',       linestyle='-', label='iGPU-LAIKA',color='red')          # Dashed line without points
#ax.plot(batch, gpu_total,   marker='x', linestyle='-', label='GPU ‑ Total')        # Solid line with points

# Advantage range: APU 1‑512, GPU 1024‑4096
# ax.axvspan(8, 256,   color='lightgreen', alpha=0.5, label='Per‑Launch Kernel Advantage')
# ax.axvspan(512, 4096, color='lightblue',  alpha=0.5, label='Persistent Kernel Advantage')

# # Annotation example (can be adjusted as needed)
# ax.annotate('Persistent Kernel faster\n(~2.3× vs Per‑Launch Kernel)',
#             xy=(12, 26), xytext=(20, 80),
#             arrowprops=dict(arrowstyle='->', lw=0.7),
#             fontsize=14)
# 
# Add value display for the leftmost point
ax.annotate(f'{apu_optimal[0]}', xy=(batch[0], apu_optimal[0]), 
            xytext=(batch[0]-0.25, apu_optimal[0]-2), fontsize=12,color='red'
            )
ax.annotate(f'{KML_dGPU[0]}', xy=(batch[0], KML_dGPU[0]), 
            xytext=(batch[0]-0.35, KML_dGPU[0]+13), fontsize=12,color='green'
            )
ax.annotate(f'{dGPU_4090[0]}', xy=(batch[0], dGPU_4090[0]), 
            xytext=(batch[0]-0.35,KML_dGPU[0]-8), fontsize=12,color='#34431A'
            )           
ax.annotate(f'{KML_CPU_batch[0]}', xy=(batch[0], KML_CPU_batch[0]), 
            xytext=(batch[0]-0.35, KML_CPU_batch[0]+30), fontsize=12,color='#1F7DB7'
            )

# Add value display for the rightmost point
ax.annotate(f'{apu_optimal[-1]}', xy=(batch[-1], apu_optimal[-1]), 
xytext=(batch[-1]-900, apu_optimal[-1]+10), fontsize=12,color='red'
            )
ax.annotate(f'{KML_dGPU[-1]}', xy=(batch[-1], KML_dGPU[-1]), 
xytext=(batch[-1]-900, KML_dGPU[-1]+10), fontsize=12, color='green'
            )
ax.annotate(f'{dGPU_4090[-1]}', xy=(batch[-1], dGPU_4090[-1]), 
xytext=(batch[-1]-900, dGPU_4090[-1]-35), fontsize=12, color='#34431A'
            )

# Advantage range: automatically detect crossover point
ax.axvspan(0, crossover_batch, color='pink', alpha=0.5, label='iGPU sweet spot')
#ax.axvspan(1024, 4096, color='lightblue',  alpha=0.5, label='GPU Advantage')

# Axes and title
ax.set_xscale('log', base=2)
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT+8')
ax.set_xlabel(f'Generated at: {current_time}', fontsize=14)
ax.set_ylabel('Time (us)', fontsize=14)
ax.set_ylim(0, 350)

ax.grid(True, which='both', linestyle=':')
ax.legend(loc='upper left', fontsize=12)

plt.tight_layout()
plt.savefig('./result.pdf', format='pdf', bbox_inches='tight')
plt.close()
