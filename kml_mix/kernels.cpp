#include <hip/hip_runtime.h>
#include <stdio.h>
#include <string.h>
#include "weights.h"

__global__ void matrix_mult_constant(float *src, float constant, float *dest) {
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int dim = blockDim.x;
    dest[blockId*dim + threadId] = src[blockId*dim + threadId] * constant;
}

__global__ void matrix_add(float *src, float *add, float *dest) {
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int dim = blockDim.x;
    dest[blockId*dim + threadId] = src[blockId*dim + threadId] + add[blockId*dim + threadId];
}

__global__ void matrix_div_constant(float *src, float constant, float *dest) {
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int dim = blockDim.x;
    dest[blockId*dim + threadId] = src[blockId*dim + threadId] / constant;
}

__global__ void set_matrix_with_matrix(float *src, float *dest) { 
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int dim = blockDim.x;
    dest[blockId*dim + threadId] = src[blockId*dim + threadId];
}

__global__ void matrix_sub(float *src, float *sub, float *dest) { 
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int dim = blockDim.x;
    dest[blockId*dim + threadId] = src[blockId*dim + threadId] - sub[blockId*dim + threadId];
}

__global__ void matrix_elementwise_mult(float *m1, float *m2, float *dest) { 
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int dim = blockDim.x;
    dest[blockId*dim + threadId] = m1[blockId*dim + threadId] * m2[blockId*dim + threadId];
}

__global__ void matrix_elementwise_div(float *m1, float *m2, float *dest) { 
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int dim = blockDim.x;
    dest[blockId*dim + threadId] = m1[blockId*dim + threadId] / m2[blockId*dim + threadId];
}

__global__ void matrix_map(float *src, float *dest) { 
    int threadId = threadIdx.x;
    float r = 1;
    float x = src[threadId];
    long long i = *(long long *)&x;
    i = 0x5fe6eb50c7b537a9 - (i >> 1);
    r = *(float *)&i;
    r = r * (1.5f - 0.5f * x * r * r);
    r = r * (1.5f - 0.5f * x * r * r);
    r = r * (1.5f - 0.5f * x * r * r);
    r = r * (1.5f - 0.5f * x * r * r);
    r = r * (1.5f - 0.5f * x * r * r);
    dest[threadId] = r * x;
}

/*__global__ void matrix_transpose(float *m, float *ret, int rows_ret, int cols_ret) { 
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    ret[blockId*cols_ret + threadId] = m[threadId * rows_ret + blockId];
}*/

__global__ void matrix_repmat(float *m, int row_repeat, int col_repeat, int m_rows, int m_cols, float *ret) { 
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int dim = blockDim.x;
    if (col_repeat > 1) {
        for (int col_copy = 0; col_copy < col_repeat *m_cols; col_copy += m_cols) {
            ret[blockId*dim + (threadId +col_copy )] = m[blockId*dim + threadId];
        }
    }else {
        ret[blockId*dim + threadId] = m[blockId*dim + threadId];
    }
    if(row_repeat > 1) {
        for (int row_copy = m_rows; row_copy < m_rows*row_repeat; row_copy += m_rows) { 
            ret[(row_copy + blockId)*dim + threadId] = m[blockId*dim + threadId];
        }
    }
}

/*__global__ void matrix_mult(float *a, float *b, float *c, int m, int n, int k) { 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < k && row < m) {
        for(int i = 0; i < n; i++) {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}*/

__global__ void get_average(float *avg_init, int k1, int k2, int batch_size, float *inp, float * avg_final) {
    int threadId = threadIdx.x;
    float sum = 0;
    for(int j = 0; j < batch_size; j++) {
        sum += inp[j*5 + threadId];
    }
    avg_final[threadId] = avg_init[threadId] * k1 /(k2  + batch_size)
        + sum / (k2 + batch_size);
}

__global__ void get_variance(float *var_init, float k1, float k2, 
        int batch_size ,float *inp, float *data_last_values, float * var_final) {
    int threadId = threadIdx.x;
    float sum_diff = 0;
    for(int j = 0; j < batch_size; j++) {
        sum_diff += (data_last_values[threadId] -  inp[ j*5 + threadId])
        *(data_last_values[threadId] -  inp[j*5 + threadId]) ;
    }
    var_final[threadId] = var_init[threadId] * k1 /(k2 + batch_size) 
        + sum_diff/ (k2 + batch_size);
}

__global__ void normalize_data(float *inp, float *avg, float *std_dev, float *out) {
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int dim = blockDim.x;
    out[blockId * dim + threadId] = (inp[blockId * dim + threadId] - avg[threadId])
        / std_dev[threadId];
}

__global__ void add_bias(float *wx, float *bias, float *out) {
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int dim = blockDim.x;
    out[blockId * dim + threadId] = wx[blockId * dim + threadId] + bias[threadId];
}

__global__ void matrix_argmax(float *src, int cols, int *max_col_array) {
    int threadId = threadIdx.x;
    int max_col = 0;
    int max = INT_MIN;
    for(int i = 0; i < cols; i++) {
        if(max < src[threadId * cols + i]) {
            max = src[threadId * cols + i];
            max_col = i;
        }
    }
    max_col_array[threadId] = max_col;
}

__global__ void normalize_fused(int batch_size, float* inputs, float* avg_base, float* avg_out, 
        float* last_values, float* var_out, float* final_out) {
    int tid = threadIdx.x;
    int uid = blockIdx.x * blockDim.x + threadIdx.x;
    int intrablock_idx = tid % 5;
    int inputs_per_block = 32;
    int base_idx = blockIdx.x * inputs_per_block + intrablock_idx;

    float* input  = inputs    + base_idx;
    float* output = final_out + base_idx;

    const int MAX = 5*batch_size;
    const int k1 = 10;
    const int k2 = 9;

    if (uid < 5) {
        float sum = 0;
        for(int j = 0; j < batch_size; j++) {
            sum += inputs[j*5 + tid];
        }
        avg_out[uid] = avg_base[uid] * k1 /(k2  + batch_size) + sum / (k2 + batch_size);
    }

    if (uid > 15 && uid <= 20) {
        float* var_init = avg_base;
        int idx = uid - 16;

        float sum_diff = 0;
        for(int j = 0; j < batch_size; j++) {
            sum_diff += (last_values[idx] -  inputs[j*5 + idx]) * (last_values[idx] -  inputs[j*5 + idx]) ;
        }
        var_out[idx] = var_init[idx] * k1 /(k2 + batch_size) + sum_diff/ (k2 + batch_size);

        float r = 1;
        float x = var_out[idx];
        long long i = *(long long *)&x;
        i = 0x5fe6eb50c7b537a9 - (i >> 1);
        r = *(float *)&i;
        r = r * (1.5f - 0.5f * x * r * r);
        r = r * (1.5f - 0.5f * x * r * r);
        r = r * (1.5f - 0.5f * x * r * r);
        r = r * (1.5f - 0.5f * x * r * r);
        r = r * (1.5f - 0.5f * x * r * r);
        var_out[idx] =  r * x;
    }

    __syncthreads();

    if (uid < MAX) {
        *output = (*input - avg_out[intrablock_idx]) / var_out[intrablock_idx];
    }
}

__global__ void fused_forward(float *input, int* result, int batch_size, 
        float* d_w0, float* d_b0, float* wt0,
        float* d_w1, float* d_b1, float* wt1,
        float* d_w2, float* d_b2, float* wt2,
        float* d_out0, float* d_out1, float* d_out2) {
    int tid = threadIdx.x;

    {
        float* my_row = input + blockIdx.x * 5;
        float* my_out = d_out0 + blockIdx.x * 15;

        if (tid < 15) {
            float acc = 0;
            for(int i = 0; i < 5; i++) {
                acc += my_row[i] * wt0[i*5 + tid];
            }
            my_out[tid] = acc + d_b0[tid];
        }
    }

    __syncthreads();

    {
        float* my_row = d_out0 + blockIdx.x * 15;
        float* my_out = d_out1 + blockIdx.x * 5;

        if (tid < 5) {
            float acc = 0;
            for(int i = 0; i < 15; i++) {
                acc += my_row[i] * wt1[i*15 + tid];
            }
            my_out[tid] = acc + d_b1[tid];
        }
    }

    __syncthreads();

    {
        float* my_row = d_out1 + blockIdx.x * 5;
        float* my_out = d_out2 + blockIdx.x * 4;

        if (tid < 4) {
            float acc = 0;
            for(int i = 0; i < 15; i++) {
                acc += my_row[i] * wt1[i*15 + tid];
            }
            my_out[tid] = acc + d_b2[tid];
        }
    }

    __syncthreads();

    {
        float* my_row = d_out2 + blockIdx.x * 4;
        if (tid == 0) {
            int idx = 0;
            for(int i = 1; i < 3; i++) {
                if (my_row[i] > my_row[idx])
                    idx = i;    
            }
            result[blockIdx.x] = idx;
        }
    }
}


__global__ void normalize_fused_persistent(int batch_size, float* inputs, float* avg_base, float* avg_out, 
    float* last_values, float* var_out, float* final_out, int* task_flag, int* quit_flag) {
        int tid = threadIdx.x;
        int uid = blockIdx.x * blockDim.x + threadIdx.x;
        int intrablock_idx = tid % 5;
        int inputs_per_block = 32;
        int base_idx = blockIdx.x * inputs_per_block + intrablock_idx;
    
        float* input  = inputs    + base_idx;
        float* output = final_out + base_idx;
    
        const int MAX = 5*batch_size;
        const int k1 = 10;
        const int k2 = 9;
        
        
        while (true) {//persistent kernel 1
        __threadfence_system();
        if (*quit_flag) break;
        if(*task_flag == 1) {
        // 等待同步标志
        if (uid < 5) {  
            float sum = 0;
            for(int j = 0; j < batch_size; j++) {
                sum += inputs[j*5 + tid];
            }
            avg_out[uid] = avg_base[uid] * k1 /(k2  + batch_size) + sum / (k2 + batch_size);
        }
    
        if (uid > 15 && uid <= 20) {
            float* var_init = avg_base;
            int idx = uid - 16;
    
            float sum_diff = 0;
            for(int j = 0; j < batch_size; j++) {
                sum_diff += (last_values[idx] -  inputs[j*5 + idx]) * (last_values[idx] -  inputs[j*5 + idx]) ;
            }
            var_out[idx] = var_init[idx] * k1 /(k2 + batch_size) + sum_diff/ (k2 + batch_size);
    
            float r = 1;
            float x = var_out[idx];
            long long i = *(long long *)&x;
            i = 0x5fe6eb50c7b537a9 - (i >> 1);
            r = *(float *)&i;
            r = r * (1.5f - 0.5f * x * r * r);
            r = r * (1.5f - 0.5f * x * r * r);
            r = r * (1.5f - 0.5f * x * r * r);
            r = r * (1.5f - 0.5f * x * r * r);
            r = r * (1.5f - 0.5f * x * r * r);
            var_out[idx] =  r * x;
        }
    
        __syncthreads();
    
        if (uid < MAX) {
            *output = (*input - avg_out[intrablock_idx]) / var_out[intrablock_idx];
        }
    
        // 设置完成标志
            if (tid == 0 && blockIdx.x == 0) {
                *task_flag = 2; // 表示 normalize 完成
                }
            }
            else{
                __builtin_amdgcn_s_sleep(10000);        
                }
        }
}

__global__ void fused_forward_persistent(float *input, int* result, int batch_size, 
    float* d_w0, float* d_b0, float* wt0,
    float* d_w1, float* d_b1, float* wt1,
    float* d_w2, float* d_b2, float* wt2,
    float* d_out0, float* d_out1, float* d_out2, int* task_flag, int* quit_flag) {
int tid = threadIdx.x;



while (true) { // persistent kernel 循环
    __threadfence_system();
    if (*quit_flag) break;
    
    if (*task_flag == 2) { // 等待 normalize 完成
        // 第一阶段：第一层线性变换
        {
            float* my_row = input + blockIdx.x * 5;
            float* my_out = d_out0 + blockIdx.x * 15;

            if (tid < 15) {
                float acc = 0;
                for(int i = 0; i < 5; i++) {
                    acc += my_row[i] * wt0[i*5 + tid];
                }
                my_out[tid] = acc + d_b0[tid];
            }
        }

        __syncthreads();

        // 第二阶段：第二层线性变换
        {
            float* my_row = d_out0 + blockIdx.x * 15;
            float* my_out = d_out1 + blockIdx.x * 5;

            if (tid < 5) {
                float acc = 0;
                for(int i = 0; i < 15; i++) {
                    acc += my_row[i] * wt1[i*15 + tid];
                }
                my_out[tid] = acc + d_b1[tid];
            }
        }

        __syncthreads();

        // 第三阶段：第三层线性变换
        {
            float* my_row = d_out1 + blockIdx.x * 5;
            float* my_out = d_out2 + blockIdx.x * 4;

            if (tid < 4) {
                float acc = 0;
                for(int i = 0; i < 5; i++) {
                    acc += my_row[i] * wt2[i*4 + tid];
                }
                my_out[tid] = acc + d_b2[tid];
            }
        }

        __syncthreads();

        // 第四阶段：argmax 操作
        {
            float* my_row = d_out2 + blockIdx.x * 4;
            if (tid == 0) {
                int idx = 0;
                for(int i = 1; i < 3; i++) {
                    if (my_row[i] > my_row[idx])
                        idx = i;    
                }
                result[blockIdx.x] = idx;
            }
        }

        // 设置完成标志
        if (tid == 0 && blockIdx.x == 0) {
            *task_flag = 3; // 表示 forward 完成
        }
    } 
    else {
        __builtin_amdgcn_s_sleep(10000); // 休眠等待
       
    }
}
}




