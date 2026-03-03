import json

file_path = '/Users/richard/Git/LLMs-from-scratch/ch03/03_understanding-buffers/understanding-buffers.ipynb'
output_path = '/Users/richard/Git/LLMs-from-scratch/ch03/03_understanding-buffers/understanding-buffers_zh.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def set_cell(idx, text):
    if idx < len(nb['cells']):
        nb['cells'][idx]['source'] = [line + '\n' for line in text.split('\n')]
        if nb['cells'][idx]['source']:
            nb['cells'][idx]['source'][-1] = nb['cells'][idx]['source'][-1].rstrip('\n')

set_cell(0, '<table style="width:100%">\n<tr>\n<td style="vertical-align:middle; text-align:left;">\n<font size="2">\n<a href="http://mng.bz/orYv">Build a Large Language Model From Scratch</a> 书籍的补充代码，作者 <a href="https://sebastianraschka.com">Sebastian Raschka</a><br>\n<br>代码仓库：<a href="https://github.com/rasbt/LLMs-from-scratch">https://github.com/rasbt/LLMs-from-scratch</a>\n</font>\n</td>\n<td style="vertical-align:middle; text-align:left;">\n<a href="http://mng.bz/orYv"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/cover-small.webp" width="100px"></a>\n</td>\n</tr>\n</table>')
set_cell(1, '# 理解 PyTorch 缓冲区')
set_cell(2, '本质上，PyTorch 缓冲区是与 PyTorch 模块或模型关联的张量属性，类似于参数，但与参数不同的是，缓冲区在训练期间不会更新。\n\nPyTorch 中的缓冲区在处理 GPU 计算时特别有用，因为它们需要与模型的参数一起在设备之间传输（例如从 CPU 到 GPU）。与参数不同，缓冲区不需要梯度计算，但它们仍然需要位于正确的设备上以确保所有计算正确执行。\n\n在第 3 章中，我们通过 `self.register_buffer` 使用 PyTorch 缓冲区，书中对此只做了简要说明。由于概念和目的不是立即清晰的，本代码笔记本通过一个动手示例提供了更详细的解释。')
set_cell(3, '## 一个没有缓冲区的例子')
set_cell(4, '假设我们有以下代码，它是基于第 3 章的代码。此版本已修改为不包含缓冲区。它实现了 LLM 中使用的因果自注意力机制：')
set_cell(6, '我们可以对一些示例数据初始化并运行该模块，如下所示：')
set_cell(8, '到目前为止，一切正常。\n\n然而，在训练 LLM 时，我们通常使用 GPU 来加速过程。因此，让我们将 `CausalAttentionWithoutBuffers` 模块转移到 GPU 设备上。\n\n请注意，此操作要求代码在配备 GPU 的环境中运行。')
set_cell(10, '现在，让我们再次运行代码：')
set_cell(12, '运行代码导致了错误。发生了什么？看来我们尝试在 GPU 上的张量和 CPU 上的张量之间进行矩阵乘法。但我们已经将模块移动到了 GPU！？\n\n\n让我们仔细检查一些张量的设备位置：')
set_cell(15, '正如我们所见，`mask` 没有被移动到 GPU 上。这是因为它不是像权重（例如 `W_query.weight`）那样的 PyTorch 参数。\n\n这意味着我们必须通过 `.to("cuda")` 手动将其移动到 GPU：')
set_cell(17, '让我们再次尝试代码：')
set_cell(19, '这次成功了！\n\n然而，记住将单个张量移动到 GPU 可能是乏味的。正如我们在下一节将看到的，使用 `register_buffer` 将 `mask` 注册为缓冲区会更容易。')
set_cell(20, '## 一个带有缓冲区的例子')
set_cell(21, '现在让我们修改因果注意力类，将因果 `mask` 注册为缓冲区：')
set_cell(23, '现在，方便的是，如果我们移动模块到 GPU，mask 也会位于 GPU 上：')
set_cell(25, '正如我们在上面看到的，将张量注册为缓冲区可以让我们的生活轻松很多：我们不必记住手动将张量移动到目标设备（如 GPU）。')
set_cell(26, '## 缓冲区和 `state_dict`')
set_cell(27, '- PyTorch 缓冲区的另一个优势（相对于普通张量）是它们会被包含在模型的 `state_dict` 中\n- 例如，考虑没有缓冲区的因果注意力对象的 `state_dict`')
set_cell(29, '- 上面的 `state_dict` 中不包含 mask\n- 然而，mask *确实* 包含在下面的 `state_dict` 中，这要归功于将其注册为缓冲区')
set_cell(31, '- `state_dict` 在保存和加载训练好的 PyTorch 模型时非常有用\n- 在这种特定情况下，保存和加载 `mask` 可能不是非常有用，因为它在训练期间保持不变；所以，为了演示目的，让我们假设它被修改了，其中所有的 `1` 都被更改为 `2`：')
set_cell(33, '- 然后，如果我们保存并加载模型，我们可以看到 mask 恢复了修改后的值')
set_cell(35, '- 如果我们不使用缓冲区，这就不是真的：')

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
