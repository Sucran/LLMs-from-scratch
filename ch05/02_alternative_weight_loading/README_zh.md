# 加载预训练权重的替代方法

此文件夹包含替代权重加载策略，以防无法从 OpenAI 获取权重。

- [weight-loading-pytorch.ipynb](weight-loading-pytorch.ipynb)：（推荐）包含从 PyTorch 状态字典加载权重的代码，这些状态字典是我通过转换原始 TensorFlow 权重创建的

- [weight-loading-hf-transformers.ipynb](weight-loading-hf-transformers.ipynb)：包含通过 `transformers` 库从 Hugging Face Model Hub 加载权重的代码

- [weight-loading-hf-safetensors.ipynb](weight-loading-hf-safetensors.ipynb)：包含直接通过 `safetensors` 库从 Hugging Face Model Hub 加载权重的代码（跳过 Hugging Face transformer 模型的实例化）
