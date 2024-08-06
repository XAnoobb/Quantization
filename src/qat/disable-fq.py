import torch
import torchvision
from pytorch_quantization import tensor_quant
from pytorch_quantization import quant_modules
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from pytorch_quantization import calib
from typing import List, Callable, Union, Dict

# 定义一个禁用量化类
class disable_quantization:
    def __init__(self, model):
        self.model = model

    def apply(self, disabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = disabled

    def __enter__(self):
        self.apply(True)    
    
    def __exit__(self, *args, **kwargs):
        self.apply(False)


# 定义一个启用量化类
class enable_quantization:
    def __init__(self, model):
        self.model = model
    
    def apply(self, enabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = not enabled
    
    def __enter__(self):
        self.apply(True)
        return self
    
    def __exit__(self, *args, **kwargs):
        self.apply(False)

# 打印某个节点的量化器状态
def quantizer_state(module):
    for name, module in module.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            print(name, module)

# 初始化量化模块
quant_modules.initialize()  # 对整个模型进行量化
# 加载预训练模型
model = torchvision.models.resnet50(pretrained=True)
# 将模型移动到GPU
model.cuda()

# 使用disable_quantization类关闭某个节点的量化
with disable_quantization(model.conv1):
    # 开启某个节点的量化
    # with enable_quantization(model.conv1):
    inputs = torch.randn(1, 3, 224, 224, device='cuda')
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    # 将模型导出为ONNX格式
    torch.onnx.export(model, inputs, 'quant_resnet50_disabelconv1.onnx', opset_version=13)
