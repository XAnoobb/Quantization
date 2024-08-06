from pytorch_quantization import tensor_quant
import torch

# 设置随机种子，保证结果可复现
torch.manual_seed(123456) 
# 生成一个长度为10的tensor
x = torch.rand(10)
# 使用tensor_quant.fake_tensor_quant函数对x进行量化，x.abs().max()获取x的最大值
fake_x = tensor_quant.fake_tensor_quant(x, x.abs().max()) # FQ算子
# 打印x
print(x)
# 打印fake_x
print(fake_x)
