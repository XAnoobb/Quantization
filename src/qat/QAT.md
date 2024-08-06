fq 基础fq算子方法


auto-fq 自动插入fq算子


disable-fq 将部分网络层的fq算子去掉


self-fq 自定义插入fq算子


nv-workflow nvidia提供的量化流程 

https://github.com/NVIDIA/TensorRT/blob/release/8.6/tools/pytorch-quantization/examples/torchvision/classification_flow.py

PTQ
-
PTQ 量化不需要训练，只需要提供一些样本图片，然后在已经训练好的模型上进行校准，统计出来需要的每一层的 scale 就可以实现量化了，大概流程如下：

1.在准备好的校准数据集上评估预训练模型
2.使用校准数据来校准模型(校准数据可以是训练集的子集)
3.计算网络中权重和激活的动态范围用来算出量化参数(q-params)
4.使用 q-params 量化网络并执行推理
具体使用就是我们导出 ONNX 模型，转换为 engine 的过程中使用 tensorRT 提供的 Calibration 方法去校准，这个使用起来比较简单。可以直接使用 tensorRT 官方提供的 trtexec 工具去实现，也可以使用它提供的 Python 或者 C++ 的 API 接口去实现。

QAT
-
QAT(Quantization Aware Training) 即训练中量化也叫显式量化。它是 tensorRT8 的一个新特性，这个特性其实是指 tensorRT 有直接加载 QAT 模型的能力。而 QAT 模型在这里是指包含 QDQ 操作的量化模型，而 QDQ 操作就是指量化和反量化操作。

实际上 QAT 过程和 tensorRT 没有太大关系，tensorRT 只是一个推理框架，实际的训练中量化操作一般都是在训练框架中去做，比如我们熟悉的 Pytorch。(当然也不排除之后一些推理框架也会有训练功能，因此同样可以在推理框架中做)

tensorRT-8 可以显式地加载包含有 QAT 量化信息的 ONNX 模型，实现一系列优化后，可以生成 INT8 的 engine。

QAT 量化需要插入 QAT 算子且需要训练进行微调，大概流程如下

1.准备一个预训练模型
2.在模型中添加 QAT 算子
3.微调带有 QAT 算子的模型
4.将微调后模型的量化参数即 q-params 存储下来
5.量化模型执行推理


TensorRT量化
-
TensorRT 量化可分为隐式量化和显示量化两种


隐式量化


1.trt7 版本之前
2.只具备 PTQ 一种量化形式
3.各层精度不可控
4.显示量化


trt8 版本之后


1.支持带 QDQ 节点的 PTQ 以及 支持带 QDQ 节点的 QAT 两种量化形式
2.带 QDQ 节点的 PTQ 是没有进行 Finetune 的，只是插入了对应的 QDQ 节点
3.带 QDQ 节点的 QAT 是进行了 Finetune 的
4.显示量化是必须带 QDQ 节点的
5.各层精度可控

