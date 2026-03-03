import json

file_path = '/Users/richard/Git/LLMs-from-scratch/ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb'
output_path = '/Users/richard/Git/LLMs-from-scratch/ch03/02_bonus_efficient-multihead-attention/mha-implementations_zh.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def set_cell(idx, text):
    if idx < len(nb['cells']):
        nb['cells'][idx]['source'] = [line + '\n' for line in text.split('\n')]
        if nb['cells'][idx]['source']:
            nb['cells'][idx]['source'][-1] = nb['cells'][idx]['source'][-1].rstrip('\n')

set_cell(0, '<table style="width:100%">\n<tr>\n<td style="vertical-align:middle; text-align:left;">\n<font size="2">\n<a href="http://mng.bz/orYv">Build a Large Language Model From Scratch</a> 书籍的补充代码，作者 <a href="https://sebastianraschka.com">Sebastian Raschka</a><br>\n<br>代码仓库：<a href="https://github.com/rasbt/LLMs-from-scratch">https://github.com/rasbt/LLMs-from-scratch</a>\n</font>\n</td>\n<td style="vertical-align:middle; text-align:left;">\n<a href="http://mng.bz/orYv"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/cover-small.webp" width="100px"></a>\n</td>\n</tr>\n</table>')
set_cell(1, '# 比较高效的多头注意力实现')
set_cell(2, '本代码笔记本比较了在 GPT、Llama 等解码器风格 LLM 中实现因果多头注意力的不同方法。')
set_cell(4, '- 要运行本笔记本中的所有代码，请确保您更新到至少 PyTorch 2.5（FlexAttention 不包含在早期的 PyTorch 版本中）\n- 如果上面的代码单元显示的 PyTorch 版本低于 2.5，您可以通过取消注释并运行以下代码单元来升级您的 PyTorch 安装（请注意，PyTorch 2.5 需要 Python 3.9 或更高版本）\n- 有关更具体的说明和 CUDA 版本，请参阅 https://pytorch.org 上的官方安装指南')
set_cell(6, '## 1. 第 3 章中的 CausalAttention MHA 包装类')
set_cell(8, '## 2. 第 3 章中的多头注意力类')
set_cell(10, '## 3. 具有组合权重的另一种多头注意力')
set_cell(11, '- 下面的 `MultiHeadAttentionCombinedQKV` 类代码基于 [Rayed Bin Wahed](https://github.com/rasbt/LLMs-from-scratch/discussions/51) 友情分享的代码\n- `MultiHeadAttentionCombinedQKV` 类与第 3 章中使用的 `MultiHeadAttention` 类之间的主要区别在于，`MultiHeadAttentionCombinedQKV` 使用单个权重矩阵 `self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)` 而不是单独的权重矩阵：\n\n  - `self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)`\n  - `self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)`\n  - `self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)`\n\n- 这里，`self.qkv` 结合了所有三个权重矩阵 `self.W_query`、`self.W_key` 和 `self.W_value`，以便在一个步骤中执行查询、键和值计算\n- 使用 `q, k, v = qkv.unbind(0)`，我们获得单独的查询、键和值张量，然后其使用方式类似于第 3 章 `MultiHeadAttention` 类中的查询、键和值张量')
set_cell(13, '## 4. 使用 Einsum 的多头注意力\n\n- 通过 [`torch.einsum`](https://pytorch.org/docs/stable/generated/torch.einsum.html) 使用爱因斯坦求和约定实现多头注意力')
set_cell(15, '## 5. 使用 PyTorch 的缩放点积注意力和 FlashAttention 的多头注意力')
set_cell(16, '- 下面的实现使用了 PyTorch 的 [`scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 函数，它实现了一个称为 [FlashAttention](https://arxiv.org/abs/2205.14135) 的内存优化版自注意力')
set_cell(19, '## 6. 不带 FlashAttention 的 PyTorch 缩放点积注意力\n\n- 这与上面类似，只是我们通过传递显式的因果掩码来禁用 FlashAttention')
set_cell(22, '## 7. 使用 PyTorch 的 torch.nn.MultiheadAttention')
set_cell(23, '- 下面，我们使用 PyTorch 的 [torch.nn.MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) 实现')
set_cell(25, '## 8. 将 PyTorch 的 torch.nn.MultiheadAttention 与 `scaled_dot_product_attention` 一起使用')
set_cell(26, '- 将 `need_weights`（默认为 `True`）设置为 `False`，以便 `MultiheadAttention` 使用 `scaled_dot_product_attention` [根据文档](https://github.com/pytorch/pytorch/blob/71d020262793542974cf13b30f2a9099773f015c/torch/nn/modules/activation.py#L1096)\n\n```markdown\nneed_weights: If specified, returns `attn_output_weights` in addition to `attn_outputs`.\n           Set `need_weights=False` to use the optimized `scaled_dot_product_attention`\n           and achieve the best performance for MHA.\n           Default: `True`\n```')
set_cell(28, '## 9. 使用 PyTorch 的 FlexAttention\n\n- 参阅 [FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention](https://pytorch.org/blog/flexattention/) 以了解有关 FlexAttention 的更多信息\n- FlexAttention 注意事项：目前不支持 dropout\n- 这从 PyTorch 2.5 开始支持，您可以通过以下方式在 CPU 机器上安装\n\n    ```bash\n    pip install torch torchvision torchaudio\n    ```\n\n- 要在 GPU 机器上安装 PyTorch，请使用以下命令（有关更多信息，另请参阅 [pytorch.org](https://pytorch.org/) 上的安装菜单）\n\n    ```bash\n    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124\n    ```')
set_cell(32, '## 10. 快速速度比较')
set_cell(33, '### 10.1 M3 Macbook Air CPU 上的速度比较')
set_cell(43, '### 10.2 Nvidia A100 GPU 上的快速速度比较')
set_cell(55, '## 11. 可视化')
set_cell(56, '### 11.1 可视化实用函数')
set_cell(60, '### 11.2 带有预热的速度比较（Nvidia A100 GPU）（仅前向传递）')
set_cell(63, '### 11.3 带有预热的速度比较（Nvidia A100 GPU）（前向和后向传递）')
set_cell(66, '### 11.4 带有预热和编译的速度比较（Nvidia A100 GPU）（前向和后向传递）')

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
