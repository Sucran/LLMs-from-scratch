# -*- coding: utf-8 -*-
import json
import os

def translate_notebook():
    source_path = 'ch05/01_main-chapter-code/ch05.ipynb'
    target_path = 'ch05/01_main-chapter-code/ch05_zh.ipynb'
    
    print(f"Reading {source_path}...")
    with open(source_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb['cells']
    
    def set_source(idx, text_list):
        if 0 <= idx < len(cells):
            if isinstance(text_list, str):
                text_list = [text_list]
            cells[idx]['source'] = text_list

    # Cell 1
    set_source(1, ["# 第 5 章：在未标记数据上进行预训练"])
    
    # Cell 4
    set_source(4, [
        "- 在本章中，我们实现训练循环和用于基本模型评估的代码，以预训练一个 LLM\n",
        "- 在本章末尾，我们还将把 OpenAI 公开可用的预训练权重加载到我们的模型中"
    ])
    
    # Cell 6
    set_source(6, ["- 本章涵盖的主题如下所示"])
    
    # Cell 8
    set_source(8, ["&nbsp;\n", "## 5.1 评估生成文本模型"])
    
    # Cell 9
    set_source(9, [
        "- 我们首先简要回顾一下使用上一章的代码初始化 GPT 模型\n",
        "- 然后，我们讨论 LLM 的基本评估指标\n",
        "- 最后，在本节中，我们将这些评估指标应用于训练和验证数据集"
    ])
    
    # Cell 10
    set_source(10, ["&nbsp;\n", "### 5.1.1 使用 GPT 生成文本"])
    
    # Cell 11
    set_source(11, ["- 我们使用上一章的代码初始化一个 GPT 模型"])
    
    # Cell 13
    set_source(13, [
        "- 我们在上面使用了 0.1 的 Dropout，但现在训练不带 Dropout 的 LLM 相对普遍\n",
        "- 现代 LLM 也不在查询、键和值矩阵的 `nn.Linear` 层中使用偏置向量（这与早期的 GPT 模型不同），这是通过设置 `\"qkv_bias\": False` 来实现的\n",
        "- 我们将上下文长度（`context_length`）减少到仅 256 个标记，以减少训练模型所需的计算资源，而最初的 1.24 亿参数 GPT-2 模型使用了 1024 个标记\n",
        "  - 这样更多的读者就可以在他们的笔记本电脑上跟随并执行代码示例\n",
        "  - 但是，请随意将 `context_length` 增加到 1024 个标记（这不需要任何代码更改）\n",
        "  - 我们稍后也将从预训练权重加载具有 1024 `context_length` 的模型"
    ])
    
    # Cell 14
    set_source(14, [
        "- 接下来，我们使用上一章的 `generate_text_simple` 函数来生成文本\n",
        "- 此外，我们定义了两个便利函数 `text_to_token_ids` 和 `token_ids_to_text`，用于在本章中使用的标记和文本表示之间进行转换"
    ])
    
    # Cell 16
    set_source(16, [
        "- 正如我们在上面看到的，模型没有生成好的文本，因为它还没有经过训练\n",
        "- 我们如何以数字形式衡量或捕捉什么是“好文本”，以便在训练期间对其进行跟踪？\n",
        "- 下一小节介绍了计算生成输出的损失指标的方法，我们可以用它来衡量训练进度\n",
        "- 关于微调 LLM 的后续章节也将介绍衡量模型质量的其他方法"
    ])
    
    # Cell 17
    set_source(17, ["&nbsp;\n", "### 5.1.2 计算文本生成损失：交叉熵和困惑度"])
    
    # Cell 18
    set_source(18, [
        "- 假设我们有一个 `inputs` 张量，其中包含 2 个训练示例（行）的标记 ID\n",
        "- 与 `inputs` 相对应，`targets` 包含我们希望模型生成的期望标记 ID\n",
        "- 请注意，`targets` 是 `inputs` 移动了 1 个位置，正如我们在第 2 章实现数据加载器时所解释的那样"
    ])
    
    # Cell 20
    set_source(20, [
        "- 将 `inputs` 输入模型，我们获得 2 个输入示例的 logits 向量，每个示例包含 3 个标记\n",
        "- 每个标记都是一个对应于词汇表大小的 50,257 维向量\n",
        "- 应用 softmax 函数，我们可以将 logits 张量转换为包含概率分数的相同维度的张量"
    ])
    
    # Cell 22
    set_source(22, [
        "- 下图使用非常小的词汇表进行说明，概述了我们将概率分数转换回文本的方式，我们在上一章末尾讨论过这一点\n",
        "\n",
        "<img src=\"https://sebastianraschka.com/images/LLMs-from-scratch-images/ch05_compressed/02.webp\" width=500px>"
    ])
    
    # Cell 24
    set_source(24, [
        "- 正如在上一章中讨论的那样，我们可以应用 `argmax` 函数将概率分数转换为预测的标记 ID\n",
        "- 上面的 softmax 函数为每个标记生成了一个 50,257 维的向量；`argmax` 函数返回此向量中最高概率分数的位置，即给定标记的预测标记 ID"
    ])
    
    # Cell 25
    set_source(25, ["- 由于我们有 2 个输入批次，每个批次有 3 个标记，因此我们获得 2 乘 3 个预测标记 ID："])
    
    # Cell 27
    set_source(27, ["- 如果我们解码这些标记，我们会发现它们与我们希望模型预测的标记（即目标标记）有很大不同："])
    
    # Cell 29
    set_source(29, [
        "- 那是因为模型还没有经过训练\n",
        "- 为了训练模型，我们需要知道它距离正确预测（目标）有多远"
    ])
    
    # Cell 31
    set_source(31, ["- 对应于目标索引的标记概率如下："])
    
    # Cell 33
    set_source(33, [
        "- 我们希望最大化所有这些值，使它们接近概率 1\n",
        "- 在数学优化中，最大化概率分数的对数比最大化概率分数本身更容易；这超出了本书的范围，但我在这里录制了一个包含更多细节的讲座：[L8.2 逻辑回归损失函数](https://www.youtube.com/watch?v=GxJe0DZvydM)"
    ])
    
    # Cell 35
    set_source(35, ["- 接下来，我们计算平均对数概率："])
    
    # Cell 37
    set_source(37, [
        "- 目标是通过优化模型权重使这个平均对数概率尽可能大\n",
        "- 由于对数的原因，最大可能值为 0，而我们目前离 0 还很远"
    ])
    
    # Cell 38
    set_source(38, [
        "- 在深度学习中，与其最大化平均对数概率，不如最小化*负*平均对数概率值；在我们的例子中，与其最大化 -10.7722 使其接近 0，在深度学习中，我们将最小化 10.7722 使其接近 0\n",
        "- -10.7722 的负值，即 10.7722，在深度学习中也被称为交叉熵损失"
    ])
    
    # Cell 40
    set_source(40, ["- PyTorch 已经实现了一个 `cross_entropy` 函数来执行前面的步骤"])
    
    # Cell 42
    set_source(42, ["- 在应用 `cross_entropy` 函数之前，让我们检查 logits 和 targets 的形状"])
    
    # Cell 44
    set_source(44, ["- 对于 PyTorch 中的 `cross_entropy` 函数，我们希望通过在批次维度上组合它们来展平这些张量："])
    
    # Cell 46
    set_source(46, [
        "- 请注意，targets 是标记 ID，它们也表示我们希望最大化的 logits 张量中的索引位置\n",
        "- PyTorch 中的 `cross_entropy` 函数将自动处理对 logits 中那些要最大化的标记索引应用 softmax 和对数概率计算"
    ])
    
    # Cell 48
    set_source(48, [
        "- 与交叉熵损失相关的一个概念是 LLM 的困惑度\n",
        "- 困惑度仅仅是交叉熵损失的指数"
    ])
    
    # Cell 50
    set_source(50, [
        "- 困惑度通常被认为更具可解释性，因为它可以理解为模型在每一步都不确定的有效词汇量大小（在上面的例子中，那是 48,725 个单词或标记）\n",
        "- 换句话说，困惑度提供了一种衡量模型预测的概率分布与数据集中单词的实际分布匹配程度的方法\n",
        "- 与损失类似，较低的困惑度表明模型预测更接近实际分布"
    ])
    
    # Cell 51
    set_source(51, ["&nbsp;\n", "### 5.1.3 计算训练集和验证集的损失"])
    
    # Cell 52
    set_source(52, [
        "- 我们使用一个相对较小的数据集来训练 LLM（实际上，只有一个短篇故事）\n",
        "- 原因是：\n",
        "  - 您可以在没有合适 GPU 的笔记本电脑上几分钟内运行代码示例\n",
        "  - 训练完成得相对较快（几分钟而不是几周），这对于教育目的很有好处\n",
        "  - 我们使用公共领域的文本，可以将其包含在此 GitHub 存储库中，而不会违反任何使用权或使存储库大小膨胀\n",
        "\n",
        "- 例如，Llama 2 7B 需要在 A100 GPU 上进行 184,320 个 GPU 小时的训练，才能在 2 万亿个标记上进行训练\n",
        "  - 在撰写本文时，AWS 的 8xA100 云服务器的小时成本约为 30 美元\n",
        "  - 因此，通过粗略计算，训练这个 LLM 将花费 184,320 / 8 * 30 美元 = 690,000 美元\n",
        "\n",
        "- 下面，我们使用与第 2 章相同的数据集"
    ])
    
    # Cell 55
    set_source(55, ["- 快速检查文本是否加载正常，打印前 99 个字符和最后 99 个字符"])
    
    # Cell 59
    set_source(59, ["- 只有 5,145 个标记，这对于训练 LLM 来说非常短，但同样，这是为了教育目的（我们稍后也将加载预训练权重）"])
    
    # Cell 60
    set_source(60, [
        "- 接下来，我们将数据集划分为训练集和验证集，并使用第 2 章的数据加载器为 LLM 训练准备批次\n",
        "- 为了可视化目的，下图假设 `max_length=6`，但对于训练加载器，我们将 `max_length` 设置为 LLM 支持的上下文长度\n",
        "- 下图为简单起见仅显示输入标记\n",
        "    - 由于我们训练 LLM 来预测文本中的下一个单词，因此目标看起来与这些输入相同，只是目标移动了一个位置\n",
        "\n",
        "<img src=\"https://sebastianraschka.com/images/LLMs-from-scratch-images/ch05_compressed/03.webp\" width=500px>"
    ])
    
    # Cell 62
    set_source(62, [
        "- 我们使用相对较小的批大小以减少计算资源需求，并且因为数据集本身就很小\n",
        "- 例如，Llama 2 7B 是以 1024 的批大小进行训练的"
    ])
    
    # Cell 63
    set_source(63, ["- 一个可选检查，确保数据已正确加载："])
    
    # Cell 65
    set_source(65, ["- 另一个可选检查，确保标记大小在预期范围内："])
    
    # Cell 67
    set_source(67, [
        "- 接下来，我们实现一个实用函数来计算给定批次的交叉熵损失\n",
        "- 此外，我们实现第二个实用函数来计算数据加载器中用户指定数量批次的损失"
    ])
    
    # Cell 69
    set_source(69, [
        "- 如果您的机器有支持 CUDA 的 GPU，LLM 将在 GPU 上训练，而无需更改代码\n",
        "- 通过 `device` 设置，我们确保将数据加载到与 LLM 模型相同的设备上"
    ])
    
    # Cell 71
    set_source(71, ["&nbsp;\n", "## 5.2 训练 LLM"])
    
    # Cell 72
    set_source(72, [
        "- 在本节中，我们终于实现了训练 LLM 的代码\n",
        "- 我们专注于一个简单的训练函数（如果您有兴趣使用更高级的技术（例如学习率预热、余弦退火和梯度裁剪）来增强此训练函数，请参阅 [附录 D](../../appendix-D/01_main-chapter-code)）"
    ])
    
    # Cell 74
    set_source(74, ["- 现在，让我们使用上面定义的训练函数来训练 LLM："])
    
    # Cell 76
    set_source(76, [
        "- 请注意，您在计算机上获得的损失值可能会略有不同，如果它们大致相似（训练损失低于 1，验证损失低于 7），则不必担心\n",
        "- 微小的差异通常可能是由于不同的 GPU 硬件和 CUDA 版本或较新 PyTorch 版本中的微小变化造成的\n",
        "- 即使您在 CPU 上运行示例，也可能会观察到微小的差异；造成差异的一个可能原因是 `nn.Dropout` 在不同操作系统上的行为不同，这取决于 PyTorch 的编译方式，详见 [PyTorch 问题追踪器](https://github.com/pytorch/pytorch/issues/121595)"
    ])
    
    # Cell 80
    set_source(80, [
        "- 观察上面的结果，我们可以看到模型一开始生成的是难以理解的单词串，而到了最后，它能够生成语法上或多或少正确的句子\n",
        "- 然而，基于训练集和验证集的损失，我们可以看到模型开始过拟合\n",
        "- 如果我们要检查它在最后写的几段话，我们会发现它们逐字逐句地包含在训练集中——它只是记住了训练数据\n",
        "- 稍后，我们将介绍可以在一定程度上减轻这种记忆的解码策略\n",
        "- 请注意，这里的过拟合发生是因为我们有一个非常非常小的训练集，并且我们对它进行了多次迭代\n",
        "  - 这里的 LLM 训练主要用于教育目的；我们主要想看看模型能否学会生成连贯的文本\n",
        "  - 我们稍后将加载预训练权重，而不是花费数周或数月在大量昂贵的硬件上训练此模型"
    ])
    
    # Cell 82
    set_source(82, ["**如果您有兴趣使用更高级的技术（例如学习率预热、余弦退火和梯度裁剪）来增强此训练函数，请参阅 [附录 D](../../appendix-D/01_main-chapter-code)**"])
    
    # Cell 83
    set_source(83, ["**如果您对更大的训练数据集和更长的训练运行感兴趣，请参阅 [../03_bonus_pretraining_on_gutenberg](../03_bonus_pretraining_on_gutenberg)**"])
    
    # Cell 84
    set_source(84, ["&nbsp;\n", "## 5.3 控制随机性的解码策略"])
    
    # Cell 85
    set_source(85, [
        "- 对于像我们上面训练的 GPT 模型这样相对较小的 LLM，推理相对便宜，因此如果您在上面使用 GPU 进行训练，则无需使用 GPU 进行推理\n",
        "- 使用我们之前在简单训练函数中使用的 `generate_text_simple` 函数（来自上一章），我们可以一次生成一个单词（或标记）的新文本\n",
        "- 正如第 5.1.2 节所解释的那样，下一个生成的标记是对应于词汇表中所有标记中最大概率分数的标记"
    ])
    
    # Cell 87
    set_source(87, [
        "- 即使我们多次执行上面的 `generate_text_simple` 函数，LLM 也总是会生成相同的输出\n",
        "- 我们现在介绍两个概念，即所谓的解码策略，以修改 `generate_text_simple`：*温度缩放*和*top-k* 采样\n",
        "- 这些将允许模型控制生成文本的随机性和多样性"
    ])
    
    # Cell 88
    set_source(88, ["&nbsp;\n", "### 5.3.1 温度缩放"])
    
    # Cell 89
    set_source(89, [
        "- 以前，我们总是使用 `torch.argmax` 采样概率最高的标记作为下一个标记\n",
        "- 为了增加多样性，我们可以使用 `torch.multinomial(probs, num_samples=1)` 采样下一个标记，从概率分布中采样\n",
        "- 这里，每个索引被选中的机会对应于它在输入张量中的概率"
    ])
    
    # Cell 90
    set_source(90, ["- 这是生成下一个标记的简要回顾，假设用于说明目的的词汇表非常小："])
    
    # Cell 92
    set_source(92, [
        "- 我们不使用 `torch.argmax` 确定最可能的标记，而是使用 `torch.multinomial(probas, num_samples=1)` 通过从 softmax 分布中采样来确定最可能的标记\n",
        "- 为了说明目的，让我们看看当我们使用原始 softmax 概率采样下一个标记 1,000 次时会发生什么："
    ])
    
    # Cell 94
    set_source(94, [
        "- 我们可以通过一个称为温度缩放的概念来控制分布和选择过程\n",
        "- “温度缩放”只是一个花哨的词，意思是将 logits 除以大于 0 的数\n",
        "- 大于 1 的温度将在应用 softmax 后导致更均匀分布的标记概率\n",
        "- 小于 1 的温度将在应用 softmax 后导致更自信（更尖锐或更峰值）的分布"
    ])
    
    # Cell 95
    set_source(95, ["- 请注意，生成的 Dropout 输出可能会因您的操作系统而异；您可以在 [PyTorch 问题追踪器](https://github.com/pytorch/pytorch/issues/121595) 上阅读有关此不一致性的更多信息"])
    
    # Cell 97
    set_source(97, ["- 我们可以看到，通过温度 0.1 进行重新缩放会导致更尖锐的分布，接近 `torch.argmax`，从而几乎总是选择最可能的单词："])
    
    # Cell 99
    set_source(99, ["- 通过温度 5 重新缩放的概率分布更加均匀："])
    
    # Cell 101
    set_source(101, ["- 假设 LLM 输入为“every effort moves you”，使用上述方法有时会导致无意义的文本，例如“every effort moves you pizza”，占 3.2%（1000 次中有 32 次）"])
    
    # Cell 102
    set_source(102, ["&nbsp;\n", "### 5.3.2 Top-k 采样"])
    
    # Cell 103
    set_source(103, ["- 为了能够使用更高的温度来增加输出多样性并减少无意义句子的概率，我们可以将采样的标记限制为前 k 个最可能的标记："])
    
    # Cell 104
    set_source(104, [
        "<img src=\"https://sebastianraschka.com/images/LLMs-from-scratch-images/ch05_compressed/15.webp\" width=500px>\n",
        "\n",
        "- （请注意，此图中的数字在小数点后截断为两位数，以减少视觉混乱。Softmax 行中的值总和应为 1.0。）"
    ])
    
    # Cell 105
    set_source(105, ["- 在代码中，我们可以按如下方式实现："])
    
    # Cell 108
    set_source(108, [
        "> 注意：\n",
        ">\n",
        ">  上一个代码单元的另一种稍微更有效的实现如下：\n",
        ">\n",
        "> ```python\n",
        "> new_logits = torch.full_like( # create tensor containing -inf values\n",
        ">    next_token_logits, -torch.inf\n",
        ">)   \n",
        "> new_logits[top_pos] = next_token_logits[top_pos] # copy top k values into the -inf tensor\n",
        "> ```\n",
        "> <br>\n",
        "> 有关更多详细信息，请参阅 https://github.com/rasbt/LLMs-from-scratch/discussions/326\n"
    ])
    
    # Cell 110
    set_source(110, ["&nbsp;\n", "### 5.3.3 修改文本生成函数"])
    
    # Cell 111
    set_source(111, [
        "- 前两个小节介绍了温度采样和 top-k 采样\n",
        "- 让我们使用这两个概念来修改第 4 章中的 `generate_text_simple` 函数，创建一个新的 `generate` 函数："
    ])
    
    # Cell 112
    set_source(112, ["&nbsp;\n", "## 5.4 在 PyTorch 中加载和保存模型权重"])
    
    # Cell 113
    set_source(113, [
        "- 训练 LLM 的计算成本很高，因此能够保存和加载 LLM 权重至关重要\n",
        "\n",
        "<img src=\"https://sebastianraschka.com/images/LLMs-from-scratch-images/ch05_compressed/16.webp\" width=400px>"
    ])
    
    # Cell 114
    set_source(114, ["- PyTorch 中推荐的方法是保存模型权重，即所谓的 `state_dict`，通过对 `.state_dict()` 方法应用 `torch.save` 函数："])
    
    # Cell 116
    set_source(116, ["- 然后，我们可以按如下方式将模型权重加载到新的 `GPTModel` 模型实例中："])
    
    # Cell 118
    set_source(118, [
        "- 使用 Adam 或 AdamW 等自适应优化器而不是常规 SGD 来训练 LLM 是很常见的\n",
        "- 这些自适应优化器为每个模型权重存储额外的参数，因此如果我们计划稍后继续预训练，将它们也保存起来是有意义的："
    ])
    
    # Cell 121
    set_source(121, ["&nbsp;\n", "## 5.5 从 OpenAI 加载预训练权重"])
    
    # Cell 122
    set_source(122, [
        "- 以前，我们只使用一本非常小的短篇故事书训练了一个小型 GPT-2 模型用于教育目的\n",
        "- 感兴趣的读者还可以在 [../03_bonus_pretraining_on_gutenberg](../03_bonus_pretraining_on_gutenberg) 中找到在完整的古腾堡计划图书语料库上进行的更长的预训练运行\n",
        "- 幸运的是，我们不必花费数万到数十万美元在大型预训练语料库上预训练模型，而是可以加载 OpenAI 提供的预训练权重"
    ])
    
    # Cell 123
    set_source(123, [
        "---\n",
        "\n",
        "---\n",
        "\n",
        "\n",
        "⚠️ **注意：由于 TensorFlow 兼容性问题，特别是某些 Windows 系统，部分用户可能会在本节遇到问题。这里需要 TensorFlow 仅是为了加载原始 OpenAI GPT-2 权重文件，然后我们将其转换为 PyTorch。\n",
        "如果您遇到与 TensorFlow 相关的问题，可以使用下面的替代代码代替本节中的剩余代码。\n",
        "此替代方案基于预转换的 PyTorch 权重，使用上一节中描述的相同转换过程创建。有关详细信息，请参阅笔记本：\n",
        "[../02_alternative_weight_loading/weight-loading-pytorch.ipynb](../02_alternative_weight_loading/weight-loading-pytorch.ipynb)。**\n",
        "\n",
        "```python\n",
        "file_name = \"gpt2-small-124M.pth\"\n"
    ])
    
    # Cell 126
    set_source(126, [
        "- 首先，一些用于从 OpenAI 下载文件并将权重加载到 Python 中的样板代码\n",
        "- 由于 OpenAI 使用了 [TensorFlow](https://www.tensorflow.org/)，我们将必须安装并使用 TensorFlow 来加载权重；[tqdm](https://github.com/tqdm/tqdm) 是一个进度条库\n",
        "- 取消注释并运行下一个单元格以安装所需的库"
    ])
    
    # Cell 130
    set_source(130, [
        "---\n",
        "\n",
        "**注意**\n",
        "\n",
        "- 在极少数情况下，上面的代码单元可能会导致 `zsh: illegal hardware instruction python` 错误，这可能是由于您机器上的 TensorFlow 安装问题\n",
        "- 一位读者发现通过 `conda` 安装 TensorFlow 解决了这个特定情况下的问题，正如 [这里](https://github.com/rasbt/LLMs-from-scratch/discussions/273#discussioncomment-12367888) 所提到的\n",
        "- 您可以在此补充 [Python 设置教程](https://github.com/rasbt/LLMs-from-scratch/tree/main/setup/01_optional-python-setup-preferences#option-2-using-conda) 中找到更多说明\n",
        "\n",
        "---\n",
        "\n",
        "- 然后我们可以按如下方式下载 124M 参数模型的模型权重："
    ])
    
    # Cell 134
    set_source(134, [
        "- 正如我们所看到的，124M GPT-2 模型具有：\n",
        "    - 50257 个词汇表大小\n",
        "    - 1024 个上下文长度\n",
        "    - 768 个嵌入大小\n",
        "    - 12 个注意力头\n",
        "    - 12 个层（Transformer 块）\n",
        "- `params` 字典包含每个层的权重张量（numpy 数组）"
    ])
    
    # Cell 136
    set_source(136, [
        "- 为了将这些权重加载到我们自己的 GPT 实现中，我们首先创建一个新的 `GPTModel` 实例\n",
        "- 请注意，我们在这里使用 `NEW_CONFIG`，因为原始 GPT-2 模型使用了偏置向量，我们之前禁用了它，并且它使用了 1024 的上下文长度（我们之前为了提高计算效率使用了 256）"
    ])
    
    # Cell 138
    set_source(138, [
        "- 现在的任务是将 OpenAI `params` 字典中的权重分配给我们 `model` 实例中的相应张量\n",
        "- 我们将为此定义一个 `load_weights_into_gpt` 函数；由于这需要大量的代码，因此将其分解为子步骤进行解释\n",
        "- 但是，在我们实现 `load_weights_into_gpt` 函数之前，让我们看看 OpenAI 和我们的实现中权重名称是如何对应的\n",
        "- 正如我们在下面看到的，权重张量名称略有不同："
    ])
    
    # Cell 141
    set_source(141, [
        "- 或者，\"355M\"、\"774M\" 和 \"1558M\" 也是受支持的 `model_size` 参数\n",
        "- 下图总结了这些不同大小的模型之间的差异："
    ])
    
    # Cell 143
    set_source(143, [
        "- 上面，我们将 124M GPT-2 模型权重加载到了 Python 中，但是我们仍然需要将它们传输到我们的 `GPTModel` 实例中\n",
        "- 首先，我们初始化一个新的 GPTModel 实例\n",
        "- 请注意，原始 GPT 模型使用偏置向量初始化多头注意力模块中的查询、键和值矩阵的线性层，这是不需要或不推荐的；但是，为了能够正确加载权重，我们也必须在我们的实现中通过将 `qkv_bias` 设置为 `True` 来启用这些\n",
        "- 我们也使用了原始 GPT-2 模型所使用的 `1024` 标记上下文长度"
    ])
    
    # Cell 146
    set_source(146, ["- 下一个任务是将 OpenAI 权重分配给我们 `GPTModel` 实例中的相应权重张量"])
    
    # Cell 149
    set_source(149, ["- 如果模型加载正确，我们可以使用它来生成新文本，使用我们之前的 `generate` 函数："])
    
    # Cell 151
    set_source(151, ["- 我们知道我们已正确加载模型权重，因为模型可以生成连贯的文本；如果我们犯了一个小错误，模型就无法做到这一点"])
    
    # Cell 152
    set_source(152, [
        "- 有关从 Hugging Face Hub 加载权重的替代方法，请参阅 [../02_alternative_weight_loading](../02_alternative_weight_loading)\n",
        "- 如果您有兴趣了解 GPT 架构与 Llama 架构（Meta AI 开发的一种流行 LLM）的比较，请参阅 [../07_gpt_to_llama](../07_gpt_to_llama) 中的奖励内容"
    ])
    
    # Cell 153
    set_source(153, ["&nbsp;\n", "## 总结和要点"])
    
    # Cell 154
    set_source(154, [
        "- 请参阅 [./gpt_train.py](./gpt_train.py) 脚本，这是一个包含训练的独立脚本\n",
        "- [./gpt_generate.py](./gpt_generate.py) 脚本从 OpenAI 加载预训练权重并根据提示生成文本\n",
        "- 您可以在 [./exercise-solutions.ipynb](./exercise-solutions.ipynb) 中找到练习解决方案"
    ])

    print(f"Writing {target_path}...")
    with open(target_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Done.")

if __name__ == "__main__":
    translate_notebook()
