import random
import numpy as np
import matplotlib.pyplot as plt
def generator_P(size):
    walk = []
    avg  = random.uniform(3.000, 600.999)
    std  = random.uniform(500.000, 1024.959)
    for _ in range(size):
        walk.append(random.gauss(avg, std))
    return walk

def smooth_distribution(p, eps=0.0001):
    is_zeros = (p == 0).astype(np.float32)
    is_nonzeros = (p != 0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros
    if not n_nonzeros:
        raise ValueError('The discrete probability distribution is malformed. All entries are 0.')
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    assert eps1 < 1.0, 'n_zeros=%d, n_nonzeros=%d, eps1=%f' % (n_zeros, n_nonzeros, eps1)
    hist = p.astype(np.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    assert (hist <= 0).sum() == 0
    return hist

import copy
import scipy.stats as stats
def threshold_distribution(distribution, target_bin=128):
    distribution = distribution[1:]
    length = distribution.size  # 获取概率分布的大小
    threshold_sum = sum(distribution[target_bin:])  # 计算概率分布从target_bin位置开始的累加和，即outliers_count
    kl_divergence = np.zeros(length - target_bin)   # 初始化一个numpy数组，用来存放每个阈值下计算得到的KL散度
    
    for threshold in range(target_bin, length):
        sliced_nd_hist = copy.deepcopy(distribution[:threshold])

        # generate reference distribution P
        p = sliced_nd_hist.copy()
        p[threshold - 1] += threshold_sum   # 将后面outliers_count加到reference_distribution_P中，得到新的概率分布  
        threshold_sum = threshold_sum - distribution[threshold] # 更新threshold_sum的值
        
        # is_nonzeros[k] indicates whether hist[k] is nonzero
        is_nonzeros = (p != 0).astype(np.int64)   # 判断每一位是否非零

        quantized_bins = np.zeros(target_bin, dtype=np.int64)
        # calculate how many bins should be merged to generate
        # quantized distribution q
        num_merged_bins = sliced_nd_hist.size // target_bin    # 计算stride
        
        # merge hist into num_quantized_bins bins
        for j in range(target_bin):
            start = j * num_merged_bins
            stop  = start + num_merged_bins
            quantized_bins[j] = sliced_nd_hist[start:stop].sum()
        quantized_bins[-1] += sliced_nd_hist[target_bin * num_merged_bins:].sum() # 将多余位累加到最后整除的位置上

        # expand quantized_bins into p.size bins
        q = np.zeros(sliced_nd_hist.size, dtype=np.float64) # 进行位扩展
        for j in range(target_bin):
            start = j * num_merged_bins
            if j == target_bin - 1:
                stop = -1
            else:
                stop = start + num_merged_bins
            norm = is_nonzeros[start:stop].sum()
            if norm != 0:
                q[start:stop] = float(quantized_bins[j]) / float(norm)

        # 平滑处理，保证KLD计算出来不会无限大
        p = smooth_distribution(p)
        q = smooth_distribution(q)

        # calculate kl_divergence between p and q
        kl_divergence[threshold - target_bin] = stats.entropy(p, q) # 计算KL散度
    
    min_kl_divergence = np.argmin(kl_divergence)    # 选择最小的KL散度
    threshold_value = min_kl_divergence + target_bin
    
    return threshold_value

if __name__ == '__main__':
    # 获取KL最小阈值
    size = 20480
    P = generator_P(size)
    P = np.array(P)
    P = P[P>0]
    print("最大的激活值", max(np.absolute(P)))

    hist, bins = np.histogram(P, bins=2048)
    threshold = threshold_distribution(hist, target_bin=128)
    print("threshold 所在组:", threshold)
    print("threshold 所在组的区间范围:", bins[threshold])
    # 分成split_zie组，density表示是否要normed
    plt.title("Relu activation value Histogram")
    plt.xlabel("Activation values")
    plt.ylabel("Normalized number of Counts")
    plt.hist(P, bins=2047)
    plt.vlines(bins[threshold], 0, 30, colors='r', linestyles='dashed')
    plt.show()
