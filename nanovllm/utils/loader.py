import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

# default_weight_loader 默认权重加载器，将加载的权重复制到模型参数中
def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)

# load_model 加载模型权重
def load_model(model: nn.Module, path: str):
    # 从 safetensors 文件加载模型权重
    # packed_modules_mapping 打包模块映射，用于处理分块权重
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    # 遍历 safetensors 文件中的每个权重
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            # 遍历每个权重
            for weight_name in f.keys():
                # 检查权重是否需要分块加载
                for k in packed_modules_mapping:
                    if k in weight_name:
                        # 获取分块索引和目标参数名
                        v, shard_id = packed_modules_mapping[k]
                        # 构建目标参数名
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break   # 关键：找到匹配后跳出循环
                # 当循环正常结束（未break）时执行，说明权重未分块加载
                else:
                    # 普通权重处理，直接加载到模型参数中
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
