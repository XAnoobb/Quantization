# 非对称量化的优点：
# 通过偏移量可以保证量化数据分布在非负数范围内，可以使得分辨率更高；
# 适合数据分布范围比较集中的情况。
# 非对称量化的缺点：
# 对于偏移量的计算需要额外的存储空间，增加了内存占用；
# 偏移量计算需要加减运算，会增加运算的复杂度；
# 对于深度学习模型，要使用int8类型的乘法指令进行计算，需要进行额外的偏置操作，增加了运算量。
import numpy as np

# 定义一个函数saturete，用于将输入的值限制在int_min和int_max之间
def saturete(x, int_max, int_min):
    return np.clip(x, int_min, int_max)

# 定义一个函数scale_z_cal，用于计算缩放比例和偏移量
def scale_z_cal(x, int_max, int_min):
    scale = (x.max() - x.min()) / (int_max - int_min)
    z = int_max - np.round((x.max() / scale))
    return scale, z

# 定义一个函数quant_float_data，用于将浮点数组量化为整数数组
def quant_float_data(x, scale, z, int_max, int_min):
    xq = saturete(np.round(x/scale + z), int_max, int_min)
    return xq

# 定义一个函数dequant_data，用于将整数数组反量化为浮点数组
def dequant_data(xq, scale, z):
    x = ((xq - z)*scale).astype('float32')
    return x

if __name__ == "__main__":
    np.random.seed(1)
    # 生成一个浮点数组
    data_float32 = np.random.randn(3).astype('float32')
    # 定义最大值和最小值
    int_max = 127
    int_min = -128
    # 打印原始数组
    print(f"input = {data_float32}")

    # 计算缩放比例和偏移量
    scale, z = scale_z_cal(data_float32, int_max, int_min)
    # 打印缩放比例和偏移量
    print(f"scale = {scale}")
    print(f"z = {z}")
    # 将浮点数组量化为整数数组
    data_int8 = quant_float_data(data_float32, scale, z, int_max, int_min)
    # 打印量化结果
    print(f"quant_result = {data_int8}")
    # 将整数数组反量化为浮点数组
    data_dequant_float = dequant_data(data_int8, scale, z)
    # 打印反量化结果
    print(f"dequant_result = {data_dequant_float}")
    
    # 打印反量化结果与原始数组的差值
    print(f"diff = {data_dequant_float - data_float32}")
