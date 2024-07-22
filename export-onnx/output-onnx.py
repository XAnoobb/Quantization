import torch
import torchvision.models as models

model = models.resnet50(pretrained=True)

input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, input, "resnet50.onnx")

# def export(
#     model: Union[torch.nn.Module, torch.jit.ScriptModule, torch.jit.ScriptFunction],
#     args: Union[Tuple[Any, ...], torch.Tensor],
#     f: Union[str, io.BytesIO],
#     export_params: bool = True,
#     verbose: bool = False,
#     training: _C_onnx.TrainingMode = _C_onnx.TrainingMode.EVAL,
#     input_names: Optional[Sequence[str]] = None,
#     output_names: Optional[Sequence[str]] = None,
#     operator_export_type: _C_onnx.OperatorExportTypes = _C_onnx.OperatorExportTypes.ONNX,
#     opset_version: Optional[int] = None,
#     do_constant_folding: bool = True,
#     dynamic_axes: Optional[
#         Union[Mapping[str, Mapping[int, str]], Mapping[str, Sequence[int]]
#     ] = None,
#     keep_initializers_as_inputs: Optional[bool] = None,
#     custom_opsets: Optional[Mapping[str, int]] = None,
#     export_modules_as_functions: Union[bool, Collection[Type[torch.nn.Module]] = False,
# )
# model: 输入的模型，可以是 torch.nn.Module、torch.jit.ScriptModule 或 torch.jit.ScriptFunction 类型。如果是非脚本化的 torch.nn.Module，函数会先将其转换为 TorchScript 图形再进行导出。
# args: 用于调用模型的输入参数。可以是元组、张量或元组+字典形式。元组中的元素可以是任意类型，包括张量。最后一个元素必须是字典，其中包含命名参数。
# f: 文件对象或文件名，用于保存导出的 ONNX 模型。
# export_params: 是否导出模型参数，默认为 True。若设为 False，导出未训练的模型。
# verbose: 是否打印详细信息，默认为 False。如果设为 True，将在标准输出打印模型描述，并启用 ONNX 导出日志记录。
# training: 模型导出时的训练模式。可选值为 _C_onnx.TrainingMode.EVAL（评估模式）、_C_onnx.TrainingMode.PRESERVE（保留当前训练状态）或 _C_onnx.TrainingMode.TRAINING（训练模式）。
# input_names: 输入节点的名字列表，按顺序指定。
# output_names: 输出节点的名字列表，按顺序指定。
# operator_export_type: 操作符导出类型。可选值为 _C_onnx.OperatorExportTypes.ONNX（导出为常规 ONNX 操作符）、_C_onnx.OperatorExportTypes.ONNX_FALLTHROUGH（尝试将所有操作符转换为默认域的 ONNX 操作符，无法转换的操作符导出为自定义域）、_C_onnx.OperatorExportTypes.ONNX_ATEN（导出为 PyTorch 自带的 ATen 实现）或 _C_onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK（尝试将 ATen 操作符转换为 ONNX 操作符，失败则导出为 ATen 操作符）。
# opset_version: ONNX 默认操作集版本，取值范围为 7 到 16。
# do_constant_folding: 是否应用常量折叠优化，默认为 True。常量折叠会替换具有全部常量输入的操作符为预计算的常量节点。
# dynamic_axes: 动态轴的字典，用于指定输入和输出张量的轴是否为动态的。键为输入或输出名称，值为轴索引和轴名称的映射或轴索引的列表。
# keep_initializers_as_inputs: 可选参数，用于决定是否将初始器作为输入。
# custom_opsets: 定义自定义操作集的映射。
# export_modules_as_functions: 将模块导出为函数的布尔值或集合。
