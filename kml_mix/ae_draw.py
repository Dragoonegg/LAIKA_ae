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

ax.plot(batch, apu_optimal, marker='o',       linestyle='-', label='iGPU-LAIKA',color='red')          # 虚线无点
#ax.plot(batch, gpu_total,   marker='x', linestyle='-', label='GPU ‑ Total')        # 实线有点

# 
# ax.axvspan(8, 256,   color='lightgreen', alpha=0.5, label='Per‑Launch Kernel Advantage')
# ax.axvspan(512, 4096, color='lightblue',  alpha=0.5, label='Persistent Kernel Advantage')

# 
# ax.annotate('Persistent Kernel faster\n(~2.3× vs Per‑Launch Kernel)',
#             xy=(12, 26), xytext=(20, 80),
#             arrowprops=dict(arrowstyle='->', lw=0.7),
#             fontsize=14)
# 
# 
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


ax.annotate(f'{apu_optimal[-1]}', xy=(batch[-1], apu_optimal[-1]), 
xytext=(batch[-1]-900, apu_optimal[-1]+10), fontsize=12,color='red'
            )
ax.annotate(f'{KML_dGPU[-1]}', xy=(batch[-1], KML_dGPU[-1]), 
xytext=(batch[-1]-900, KML_dGPU[-1]+10), fontsize=12, color='green'
            )
ax.annotate(f'{dGPU_4090[-1]}', xy=(batch[-1], dGPU_4090[-1]), 
xytext=(batch[-1]-900, dGPU_4090[-1]-35), fontsize=12, color='#34431A'
            )


ax.axvspan(0, crossover_batch, color='pink', alpha=0.5, label='iGPU sweet spot')
#ax.axvspan(1024, 4096, color='lightblue',  alpha=0.5, label='GPU Advantage')


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
