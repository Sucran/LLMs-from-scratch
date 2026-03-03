# -*- coding: utf-8 -*-
import json
import os

def translate_notebook():
    source_path = 'ch05/08_memory_efficient_weight_loading/memory-efficient-state-dict.ipynb'
    target_path = 'ch05/08_memory_efficient_weight_loading/memory-efficient-state-dict_zh.ipynb'
    
    print(f"Reading {source_path}...")
    with open(source_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb['cells']
    
    def set_source(idx, text_list):
        if 0 <= idx < len(cells):
            if isinstance(text_list, str):
                text_list = [text_list]
            cells[idx]['source'] = text_list

    # Cell 30
    set_source(30, ["# 内存高效的模型权重加载"])
    
    # Cell 38
    set_source(38, [
        "- 本笔记本提供了一些提示，用于在 GPU（或 CPU）内存有限时加载较大的预训练或微调模型\n",
        "- 具体来说，它侧重于您使用 `torch.save(model.state_dict(), \"model.pth\")` 保存模型的情况（例如，在第 5-7 章中），并希望稍后在新的会话中加载它以继续预训练或进行额外的微调\n",
        "- 虽然该示例使用 LLM，但本笔记本中解释的方法是通用的，适用于加载任何 PyTorch 模型，而不仅仅是 LLM"
    ])
    
    # Cell 89
    set_source(89, ["&nbsp;\n", "## 1. 基准测试实用程序"])
    
    # Cell 97
    set_source(97, [
        "- 首先，让我们定义一些实用程序代码来跟踪 VRAM（GPU 内存）\n",
        "- 稍后，我们还将介绍一个跟踪主系统 RAM（CPU 内存）的工具\n",
        "- 当我们稍后应用它们时，这些函数的目的将变得清晰"
    ])
    
    # Cell 143
    set_source(143, ["&nbsp;\n", "## 2. 模型设置"])
    
    # Cell 151
    set_source(151, [
        "- 此代码部分设置模型本身\n",
        "- 在这里，我们使用“大型”GPT-2 模型使事情更有趣（您可以使用“gpt2-small (124M)”来降低此笔记本的内存需求和执行时间）"
    ])
    
    # Cell 197
    set_source(197, ["- 现在，让我们看看 GPU 内存函数的实际应用："])
    
    # Cell 248
    set_source(248, ["- 此外，让我们通过传入一些示例张量来确保模型运行正常"])
    
    # Cell 274
    set_source(274, [
        "- 接下来，假设我们正在预训练模型并将其保存以供以后使用\n",
        "- 为了简单起见，我们在这里跳过实际的预训练，只保存初始化的模型（但相同的概念适用）"
    ])
    
    # Cell 298
    set_source(298, ["- 最后，我们在 Python 会话中删除模型和示例张量以重置 GPU 内存"])
    
    # Cell 332
    set_source(332, ["&nbsp;\n", "## 3. 基本权重加载"])
    
    # Cell 341
    set_source(341, [
        "- 现在开始有趣的部分，我们将加载预训练模型权重\n",
        "- 让我们看看加载之前保存的模型需要多少 GPU 内存"
    ])
    
    # Cell 387
    set_source(387, [
        "- 请注意，内存是上一个会话的 2 倍\n",
        "- 这是因为我们在短时间内在内存中拥有两次相同的模型：\n",
        "  - 第一次通过 `model.to(device)`\n",
        "  - 第二次通过代码行 `model.load_state_dict(torch.load(\"model.pth\", map_location=device, weights_only=True))`；最终，加载的模型权重将被复制到模型中，`state_dict` 将被丢弃，但在短时间内，我们在内存中同时拥有主模型和加载的 `state_dict`\n",
        "- 剩下的部分将专注于解决这个问题\n",
        "- 但首先，让我们测试模型并重置 GPU 内存\n"
    ])
    
    # Cell 432
    set_source(432, ["- 让我们测试另一种在实践中非常流行的常见模式："])
    
    # Cell 497
    set_source(497, ["- 因此，就峰值内存而言，是先在设备上实例化模型然后使用 `map_location=\"device\"`，还是先将权重加载到 CPU 内存中（`map_location=\"cpu\"`）然后再将其移动到设备上，并没有什么区别"])
    
    # Cell 507
    set_source(507, ["&nbsp;\n", "## 4. 顺序加载权重"])
    
    # Cell 516
    set_source(516, [
        "- 正如上一节中强调的那样，针对在 GPU 内存中拥有两次模型权重的问题，一个解决方法是按顺序加载模型\n",
        "- 下面，我们：\n",
        "  - 首先将模型加载到 GPU 内存中\n",
        "  - 然后将模型权重加载到 CPU 内存中\n",
        "  - 最后将每个参数一个接一个地复制到 GPU 内存中\n"
    ])
    
    # Cell 569
    set_source(569, [
        "- 正如我们在上面看到的，内存使用量比以前低得多\n",
        "- 请注意，内存从 6.4 增加到 6.7 GB，因为最初我们在内存中只有模型，然后我们在内存中有模型加上 1 个参数张量（我们暂时将参数张量移动到 GPU，以便我们可以使用 `\".to\"` 将其分配给模型）\n",
        "- 总的来说，这是一个显着的改进\n",
        "- 再次，让我们简要测试模型，然后为下一节重置 GPU 内存"
    ])
    
    # Cell 613
    set_source(613, ["&nbsp;\n", "## 5. 使用低 CPU 内存加载模型"])
    
    # Cell 622
    set_source(622, [
        "- 在上一节中，我们通过首先将权重（`state_dict`）加载到 CPU 内存中，然后再将它们一个接一个地复制到模型中，从而减少了 GPU 内存的使用\n",
        "- 但是，如果我们只有有限的 CPU 内存怎么办？\n",
        "- 本节使用 PyTorch 所谓的 `\"meta\"` 设备方法，在具有大 GPU 内存但小 CPU 内存的机器上加载模型\n",
        "- 但首先，让我们定义一个方便的函数来监控 CPU 内存"
    ])
    
    # Cell 675
    set_source(675, ["- 首先，让我们跟踪上一节中顺序权重加载方法的 CPU 内存"])
    
    # Cell 730
    set_source(730, [
        "- 现在，假设我们有一台 CPU 内存低但 GPU 内存大的机器\n",
        "- 我们可以通过引入 PyTorch 所谓的“meta”设备来权衡 CPU 内存和 GPU 内存的使用\n",
        "- PyTorch 的 meta 设备是一种特殊的设备类型，允许您在不分配实际内存的情况下创建张量，从而有效地创建“meta”张量\n",
        "- 这对于像模型分析或架构定义这样的任务很有用，因为您需要张量形状和类型，而不需要内存分配的开销"
    ])
    
    # Cell 790
    set_source(790, [
        "- 正如我们在上面看到的，通过在 meta 设备上创建模型并将权重直接加载到 GPU 内存中，我们有效地降低了 CPU 内存需求\n",
        "- 有人可能会问：“那么顺序权重加载仍然必要吗？这与原始方法相比如何？”\n",
        "- 让我们检查简单的 PyTorch 权重加载方法以进行比较（来自本笔记本中的第一个权重加载部分）："
    ])
    
    # Cell 838
    set_source(838, [
        "- 正如我们在上面看到的，没有 meta 设备的“简单”权重加载使用更多内存\n",
        "- 换句话说，如果您的机器 CPU 内存有限，您可以使用 meta 设备方法将模型权重直接加载到 GPU 内存中，以减少峰值 CPU 内存使用量"
    ])
    
    # Cell 850
    set_source(850, ["## 6. 使用 `mmap=True`（推荐）"])
    
    # Cell 858
    set_source(858, [
        "- 作为一名中级或高级 `torch.load` 用户，您可能想知道这些方法与 PyTorch 中的 `mmap=True` 设置相比如何\n",
        "- PyTorch 中的 `mmap=True` 设置启用了内存映射文件 I/O，这允许张量直接从磁盘存储访问数据，从而通过在 RAM 有限时不将整个文件加载到 RAM 中来减少内存使用\n",
        "- 另请参阅 [mikaylagawarecki](https://github.com/rasbt/LLMs-from-scratch/issues/402) 的有用评论\n",
        "- 乍一看，它可能看起来不如上面的顺序方法有效："
    ])
    
    # Cell 906
    set_source(906, [
        "- CPU RAM 使用率如此之高的原因是因为这台机器上有足够的 CPU RAM 可用\n",
        "- 但是，如果您要在 CPU RAM 有限的机器上运行此程序，`mmap` 方法将使用更少的内存"
    ])
    
    # Cell 918
    set_source(918, ["&nbsp;\n", "## 7. 其他方法"])
    
    # Cell 926
    set_source(926, [
        "- 本笔记本侧重于在 PyTorch 中加载权重的简单内置方法\n",
        "- 对于 CPU 内存有限的情况，推荐的方法是已经解释过的 `mmap=True` 方法\n",
        "- 或者，另一种选择是暴力方法，即分别保存和加载每个权重张量："
    ])

    print(f"Writing {target_path}...")
    with open(target_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Done.")

if __name__ == "__main__":
    translate_notebook()
