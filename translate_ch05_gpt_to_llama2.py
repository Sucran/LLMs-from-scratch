# -*- coding: utf-8 -*-
import json
import os

def translate_notebook():
    source_path = 'ch05/07_gpt_to_llama/converting-gpt-to-llama2.ipynb'
    target_path = 'ch05/07_gpt_to_llama/converting-gpt-to-llama2_zh.ipynb'
    
    print(f"Reading {source_path}...")
    with open(source_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb['cells']
    
    def set_source(idx, text_list):
        if 0 <= idx < len(cells):
            if isinstance(text_list, str):
                text_list = [text_list]
            cells[idx]['source'] = text_list

    # Cell 32
    set_source(32, ["# 将从零开始的 GPT 架构转换为 Llama 2"])
    
    # Cell 41
    set_source(41, [
        "- 在本笔记本中，我们将原始 GPT 架构逐步转换为 Llama 2 模型（注意 GPT 和 GPT-2 共享相同的架构）\n",
        "- 为什么不是 Llama 1 或 Llama 3？\n",
        "   - Llama 1 架构与 Llama 2 相似，只是 Llama 2 具有更大的上下文窗口（这很好）；Llama 1 权重不易获得，并且使用限制更多，因此关注 Llama 2 更合理\n",
        "   - 关于 Llama 3，我将分享一个单独的笔记本，将 Llama 2 转换为 Llama 3（只有一些小的额外更改）\n",
        "- 本笔记本中的解释故意保持最少，以免不必要地膨胀，并专注于主要代码\n",
        "- 有关更多信息，请参阅 Llama 2 论文：[Llama 2: Open Foundation and Fine-Tuned Chat Models (2023)](https://arxiv.org/abs/2307.09288)"
    ])
    
    # Cell 66
    set_source(66, ["- 本笔记本中使用的包："])
    
    # Cell 110
    set_source(110, ["&nbsp;\n", "# 1. 逐步转换 GPT 模型实现"])
    
    # Cell 121
    set_source(121, [
        "- 在本节中，我们将浏览 [第 4 章](../../ch04/01_main-chapter-code/ch04.ipynb) 中的 GPT 模型代码，并逐步修改它以实现 Llama 2 架构\n",
        "- 稍后，我们加载 Meta AI 分享的原始 Llama 2 权重"
    ])
    
    # Cell 132
    set_source(132, ["&nbsp;\n", "## 1.1 将 LayerNorm 替换为 RMSNorm 层"])
    
    # Cell 143
    set_source(143, [
        "- 首先，我们将 LayerNorm 替换为均方根层归一化 (RMSNorm)\n",
        "- LayerNorm 使用均值和方差归一化输入，而 RMSNorm 仅使用均方根，这提高了计算效率\n",
        "- The RMSNorm operation is as follows, where $x$ is the input $\\gamma$ is a trainable parameter (vector), and $\\epsilon$ is a small constant to avoid zero-division errors:\n",
        "\n",
        "$$y_i = \\frac{x_i}{\\text{RMS}(x)} \\gamma_i, \\quad \\text{where} \\quad \\text{RMS}(x) = \\sqrt{\\epsilon + \\frac{1}{n} \\sum x_i^2}$$\n",
        "\n",
        "- 有关更多详细信息，请参阅论文 [Root Mean Square Layer Normalization (2019)](https://arxiv.org/abs/1910.07467)"
    ])
    
    # Cell 203
    set_source(203, ["- 以下代码单元检查此实现是否与 PyTorch 的内置实现工作方式相同："])
    
    # Cell 232
    set_source(232, ["&nbsp;\n", "## 1.2 将 GELU 替换为 SiLU 激活"])
    
    # Cell 243
    set_source(243, [
        "- Llama 使用 SiLU 激活函数（而不是 GELU），这也称为 Swish 函数：\n",
        "\n",
        "$$\n",
        "\\text{silu}(x) = x \\cdot \\sigma(x), \\quad \\text{where} \\quad \\sigma(x) \\text{ is the logistic sigmoid.}\n",
        "$$\n",
        "\n",
        "- 有关更多信息，请参阅 SiLU 论文：[Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning (2017)](https://arxiv.org/abs/1702.03118)"
    ])
    
    # Cell 305
    set_source(305, ["&nbsp;\n", "## 1.3 更新前馈模块"])
    
    # Cell 316
    set_source(316, [
        "- 实际上，Llama 使用 SiLU 的“门控线性单元”（GLU）变体，称为 SwiGLU，这本质上导致了一个结构略有不同的 `FeedForward` 模块\n",
        "- SwiGLU 在前馈层中使用门控机制，公式为：\n",
        "\n",
        "$$\\text{SwiGLU}(x) = \\text{SiLU}(\\text{Linear}_1(x)) * (\\text{Linear}_2(x))$$\n",
        "\n",
        "- 这里，$\\text{Linear}_1$ 和 $\\text{Linear}_2$ 是两个线性层，$*$ 表示逐元素乘法\n",
        "- 第三个线性层 $\\text{Linear}_3$ 在此门控激活后应用\n",
        "\n",
        "- 有关更多信息，请参阅 SwiGLU 论文：[GLU Variants Improve Transformer (2020)](https://arxiv.org/abs/2002.05202)"
    ])
    
    # Cell 394
    set_source(394, [
        "- 请注意，我们还在上面添加了 `dtype=cfg[\"dtype\"]` 设置，这将允许我们稍后直接以较低精度格式加载模型以减少内存使用（相对于以原始 32 位精度格式实例化然后转换它）\n",
        "- 我们还设置了 `bias=False`，因为 Llama 不使用任何偏置单元"
    ])
    
    # Cell 397 (Was 394 in my notes but 397 in index)
    set_source(397, ["&nbsp;\n", "## 1.4 实现 RoPE"])
    
    # Cell 405
    set_source(405, [
        "- 在 GPT 模型中，位置嵌入实现如下：\n",
        "\n",
        "```python\n",
        "self.pos_emb = nn.Embedding(cfg[\"context_length\"], cfg[\"emb_dim\"])\n",
        "```\n",
        "\n",
        "- 与传统的绝对位置嵌入不同，Llama 使用旋转位置嵌入 (RoPE)，使其能够同时捕获绝对和相对位置信息\n",
        "- RoPE 的参考论文是 [RoFormer: Enhanced Transformer with Rotary Position Embedding (2021)](https://arxiv.org/abs/2104.09864)\n",
        "- RoPE 可以以两种等效方式实现：*split-halves* 版本和 *interleaved even/odd* 版本；只要我们一致地配对维度并使用相同的 cos/sin 排序，它们在数学上是相同的（有关更多信息，请参阅 [此](https://github.com/rasbt/LLMs-from-scratch/issues/751) GitHub 讨论）\n",
        "- 此代码使用 RoPE *split-halves* 方法，类似于 Hugging Face transformers ([modeling_llama.py](https://github.com/huggingface/transformers/blob/e42587f596181396e1c4b63660abf0c736b10dae/src/transformers/models/llama/modeling_llama.py#L173-L188))\n",
        "- 然而，原始 RoPE 论文和 Meta 的官方 Llama 2 存储库使用 *interleaved (even/odd)* 版本 ([llama/model.py](https://github.com/meta-llama/llama/blob/6c7fe276574e78057f917549435a2554000a876d/llama/model.py#L64-L74))；但如前所述，它们是等效的"
    ])
    
    # Cell 475
    set_source(475, ["- 以下是将 RoPE 应用于 `q` 和 `k` 张量的示例："])
    
    # Cell 513
    set_source(513, ["&nbsp;\n", "## 1.5 将 RoPE 添加到 MultiHeadAttention 模块"])
    
    # Cell 524
    set_source(524, [
        "- 重要的是要注意，GPT 将位置嵌入应用于输入，而 Llama 在自注意力机制本身中对查询和键向量应用旋转\n",
        "- 在这里，我们使用适当的 RoPE 代码修改 `MultiHeadAttention` 类\n",
        "- 此外，我们移除 `qkv_bias` 选项并硬编码 `bias=False` 设置\n",
        "- 此外，我们添加了一个 dtype 设置，以便稍后能够以较低精度实例化模型\n",
        " - 提示：由于 `TransformerBlock`（在下一节中）完全重复，我们可以简化代码并仅初始化缓冲区一次，而不是为每个 `MultiHeadAttention` 模块初始化；但是，我们将预先计算的 RoPE 参数添加到 `MultiHeadAttention` 类中，以便它可以作为独立模块运行"
    ])
    
    # Cell 622
    set_source(622, ["- 下面是一个在示例输入上使用 `MultiHeadAttention` 模块的示例："])
    
    # Cell 663
    set_source(663, ["&nbsp;\n", "## 1.6 更新 TransformerBlock 模块"])
    
    # Cell 674
    set_source(674, [
        "- 在这个阶段，大部分艰苦的工作已经完成；我们现在可以更新 `TransformerBlock` 以使用我们上面实现的代码\n",
        "- 这意味着我们\n",
        " - 将 LayerNorm 替换为 RMSNorm\n",
        " - 移除 dropout\n",
        " - 移除 `qkv_bias` 设置\n",
        " - 添加 `dtype` 设置"
    ])
    
    # Cell 741
    set_source(741, ["&nbsp;\n", "## 1.7 更新模型类"])
    
    # Cell 750
    set_source(750, [
        "- 您可能还记得 [第 5 章](../01_main-chapter-code/ch05.ipynb)，`TransformerBlock` 是主模型中的重复块\n",
        "- 我们的 Llama 模型几乎完成了；我们只需要更新 `TransformerBlock` 周围的模型代码\n",
        "- 这意味着我们\n",
        "  - 移除绝对位置嵌入，因为我们现在有 RoPE 嵌入\n",
        "  - 将 LayerNorm 替换为 RMSNorm\n",
        "  - 移除 dropout\n",
        "  - 添加 dtype 设置"
    ])
    
    # Cell 806
    set_source(806, ["&nbsp;\n", "## 2. 初始化模型"])
    
    # Cell 815
    set_source(815, [
        "- 模型代码现在已完成，我们准备初始化它\n",
        "- 在 [第 5 章](../01_main-chapter-code/ch05.ipynb) 中，我们使用以下配置文件指定 124M 参数的 GPT 模型："
    ])
    
    # Cell 846
    set_source(846, ["- 供参考，1.5B 参数的 GPT 模型配置如下所示："])
    
    # Cell 876
    set_source(876, ["- 同样，我们可以为 7B 模型定义一个 Llama 2 配置文件（为简单起见，我们在此忽略其他更大的模型）："])
    
    # Cell 906
    set_source(906, ["- 使用这些设置，我们现在可以初始化一个 Llama 2 7B 模型（注意这需要约 26 GB 的内存）"])
    
    # Cell 953
    set_source(953, [
        "- 如上所示，该模型包含 67 亿个参数（通常四舍五入并称为 7B 模型）\n",
        "- 此外，我们可以使用下面的代码计算此模型的内存需求："
    ])
    
    # Cell 1014
    set_source(1014, ["- 最后，如果适用，我们还可以将模型传输到 NVIDIA 或 Apple Silicon GPU："])
    
    # Cell 1045
    set_source(1045, ["&nbsp;\n", "## 3. 加载分词器"])
    
    # Cell 1054
    set_source(1054, [
        "- 在本节中，我们将加载模型的分词器\n",
        "- Llama 2 使用 Google 的 [SentencePiece](https://github.com/google/sentencepiece) 分词器而不是 OpenAI 的 [Tiktoken](https://github.com/openai/tiktoken)（但 Llama 3 使用 Tiktoken）\n",
        "- Meta AI 在 Hugging Face Hub 上分享了原始 Llama 2 模型权重和分词器词汇表\n",
        "- 我们将从 Hub 下载分词器词汇表并将其加载到 SentencePiece 中\n",
        "- 取消注释并运行以下代码以安装所需的库："
    ])
    
    # Cell 1080
    set_source(1080, [
        "- 请注意，Meta AI 要求您在下载文件之前接受 Llama 2 许可条款；为此，您必须创建一个 Hugging Face Hub 帐户并访问 [meta-llama/Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b) 存储库以接受条款\n",
        "- 接下来，您需要创建一个访问令牌；要生成具有 READ 权限的访问令牌，请单击右上角的个人资料图片，然后单击“Settings”\n",
        "\n",
        "\n",
        "<img src=\"https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gpt-to-llama/settings.webp?1\" width=\"300px\">\n",
        "\n",
        "- 然后，创建并复制访问令牌，以便您可以将其复制并粘贴到下一个代码单元中\n",
        "\n",
        "<img src=\"https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gpt-to-llama/access-token.webp?1\" width=\"600px\">"
    ])
    
    # Cell 1121
    set_source(1121, ["- 通过访问令牌登录后（这是验证我们要接受 Llama 2 许可条款所必需的），我们现在可以下载分词器词汇表："])
    
    # Cell 1167
    set_source(1167, ["- 为了为分词器提供更熟悉的接口，我们定义了一个小的 `LlamaTokenizer` 包装类："])
    
    # Cell 1205
    set_source(1205, ["- 我们现在可以使用 `generate` 函数让 Llama 2 模型生成新文本："])
    
    # Cell 1260
    set_source(1260, [
        "- 当然，正如我们在上面看到的，文本是荒谬的，因为我们还没有训练 Llama 2 模型\n",
        "- 在下一节中，我们不是自己训练它（这将花费数万到数十万美元），而是从 Meta AI 加载预训练权重"
    ])
    
    # Cell 1273
    set_source(1273, ["&nbsp;\n", "## 4. 加载预训练权重"])
    
    # Cell 1282
    set_source(1282, [
        "- 我们在下面加载 [\"meta-llama/Llama-2-7b\"](https://huggingface.co/meta-llama/Llama-2-7b) 基础模型，这是一个在微调之前的简单文本补全模型\n",
        "- 或者，您可以通过相应地修改下一个代码单元中的字符串来加载指令微调和对齐的 [\"meta-llama/Llama-2-7b-chat\"](https://huggingface.co/meta-llama/Llama-2-7b-chat) 模型"
    ])
    
    # Cell 1339
    set_source(1339, ["- `weights` 包含以下张量（为简单起见仅显示前 15 个）："])
    
    # Cell 1390
    set_source(1390, ["- 以下函数仿照 [第 5 章](../01_main-chapter-code/ch05.ipynb) 中的 `load_weights_into_gpt` 函数，将预训练权重加载到我们的 Llama 2 模型中："])
    
    # Cell 1498
    set_source(1498, ["- 接下来，我们准备使用该模型进行文本生成"])
    
    # Cell 1544
    set_source(1544, ["&nbsp;\n", "## 5. 使用指令微调模型"])
    
    # Cell 1553
    set_source(1553, ["- 如前所述，上面我们使用了预训练的基础模型；如果您想使用能够遵循指令的模型，请改用 `\"meta-llama/Llama-2-7b-chat\"` 模型，如下所示"])
    
    # Cell 1628
    set_source(1628, ["&nbsp;\n", "# 下一步是什么？"])
    
    # Cell 1635
    set_source(1635, [
        "- 本笔记本将原始 GPT-2 架构转换为 Llama 2 模型\n",
        "- 如果您有兴趣了解如何将 Llama 2 转换为 Llama 3、Llama 3.1 和 Llama 3.2，请查看 [converting-llama2-to-llama3.ipynb](converting-llama2-to-llama3.ipynb) 笔记本"
    ])

    print(f"Writing {target_path}...")
    with open(target_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Done.")

if __name__ == "__main__":
    translate_notebook()
