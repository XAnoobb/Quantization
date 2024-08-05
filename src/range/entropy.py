import numpy as np
import matplotlib.pyplot as plt

# 定义计算KL散度的函数
def cal_kl(p, q):
    KL = 0
    for i in range(len(p)):
        KL += p[i] * np.log(p[i]/q[i])
    return KL

# 定义KL检验函数
def kl_test(x, kl_threshod = 0.01):
    y_out = []
    while True:
        # 生成一个随机分布
        y = [np.random.uniform(1, size+1) for i in range(size)]
        # 计算概率分布
        y /= np.sum(y)
        # 计算KL散度
        kl_result = cal_kl(x, y)
        # 如果KL散度小于阈值，则输出结果，并画出分布图
        if kl_result < kl_threshod:
            print(kl_result)
            y_out = y
            plt.plot(x)
            plt.plot(y)
            break
    return y_out

if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(1)
    # 设置分布长度
    size = 10
    # 生成一个随机分布
    x = [np.random.uniform(1, size+1) for i in range(size)]
    # 计算概率分布
    x /= np.sum(x)
    # 进行KL检验
    y_out = kl_test(x, kl_threshod = 0.01)
    # 显示分布图
    plt.show()
    # 输出结果
    print(x, y_out)
