#对称量化的优点：
#没有偏移量，可以降低计算量
#分布在正负半轴的权值数值均可被充分利用，具有更高的利用率；
#对于深度学习模型，可以使用int8类型的乘法指令进行计算，加快运算速度；
#能够有效的缓解权值分布在不同范围内的问题。

#对称量化缺点：
#量化后的数值范围是固定的，无法处理超出该范围的数值。
#对称量化适用于权值分布均匀的情况，对于权值分布不均匀的情况，可能需要使用非对称量化。
#对于数据分布在0点附近的情况，量化的位数可能不够；

import numpy as np

# 定义一个saturete函数，用于将输入值限制在-127到127之间
def saturete(x):
    return np.clip(x, -127, 127)

# 定义一个scale_cal函数，用于计算输入值的最大绝对值
def scale_cal(x):
    max_val = np.max(np.abs(x))
    return max_val / 127

# 定义一个quant_float_data函数，用于将输入值四舍五入并缩放到-127到127之间
def quant_float_data(x, scale):
    xq = saturete(np.round(x/scale))
    return xq

# 定义一个dequant_data函数，用于将四舍五入并缩放后的值还原成原始浮点数
def dequant_data(xq, scale):
    x = (xq * scale).astype('float32')
    return x

if __name__ == "__main__":
    # 设置随机数种子
    np.random.seed(1)
    # 生成一个3个元素的浮点数数组
    data_float32 = np.random.randn(3).astype('float32')
    print(f"input = {data_float32}")

    # 计算最大绝对值
    scale = scale_cal(data_float32)
    print(f"scale = {scale}")

    # 将浮点数数组四舍五入并缩放到-127到127之间
    data_int8 = quant_float_data(data_float32, scale)
    print(f"quant_result = {data_int8}")
    # 将四舍五入并缩放后的值还原成原始浮点数
    data_dequant_float = dequant_data(data_int8, scale)
    print(f"dequant_result = {data_dequant_float}")

    # 计算还原后的浮点数与原始浮点数的差值
    print(f"diff = {data_dequant_float - data_float32}")
