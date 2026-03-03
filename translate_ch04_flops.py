# -*- coding: utf-8 -*-
import json
import os

def translate_notebook():
    source_path = 'ch04/02_performance-analysis/flops-analysis.ipynb'
    target_path = 'ch04/02_performance-analysis/flops-analysis_zh.ipynb'
    
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
    set_source(1, ["# FLOPS 分析"])
    
    # Cell 2
    set_source(2, [
        "- FLOPs（每秒浮点运算次数）通过计算执行的浮点运算数量来衡量神经网络模型的计算复杂度\n",
        "- 高 FLOPs 表示更密集的计算和能源消耗"
    ])
    
    # Cell 5
    set_source(5, ["&nbsp;\n", "# 固定批大小的简单基准测试"])
    
    # Cell 6
    set_source(6, ["- 仅前向传播"])
    
    # Cell 8
    set_source(8, ["&nbsp;\n", "# 自动寻找批大小的简单基准测试"])
    
    # Cell 9
    set_source(9, ["- 仅前向传播"])
    
    # Cell 10
    set_source(10, ["&nbsp;\n", "# 具有自动批大小查找和模型 FLOP 利用率 (MFU) 的基准测试"])
    
    # Cell 11
    set_source(11, [
        "- 模型 FLOP 利用率 (MFU) 的解释来自 [PaLM 论文](https://arxiv.org/abs/2204.02311)\n",
        "\n",
        "> 我们提出了一种新的效率指标，它与实现无关，并且允许更清晰地比较系统效率，称为模型 FLOP 利用率 (MFU)。这是观察到的吞吐量（每秒标记数）与在峰值 FLOP 下运行的系统的理论最大吞吐量之比。至关重要的是，“理论最大”吞吐量仅考虑计算前向+后向传递所需的操作，而不考虑重新具体化。\n",
        "\n",
        "\n",
        "$$\\text{MFU} = \\frac{\\text{Observed Tokens per Second}}{\\text{Theoretical Max Tokens per Second}}$$\n",
        "\n",
        "其中\n",
        "\n",
        "$$\\text{Theoretical Max Tokens per Second} = \\frac{\\text{Max FLOPs per Second}}{\\text{Total FLOPs per Token}}$$\n",
        "\n",
        "以及\n",
        "\n",
        "$$\\text{Tokens per Second} = \\frac{\\text{Batch Size} \\times \\text{Sequence Length}}{\\text{Total Time}}$$"
    ])
    
    # Cell 12
    set_source(12, ["- 前向和后向传播"])
    
    # Cell 13 - Comment in code cell
    # Skipping code modification as per general rule, but could modify comments if strictly requested.
    # The previous turn I didn't translate comments in code cells.
    
    # Cell 15
    set_source(15, [
        "- 1.0 的值是最好的（等于 100%）\n",
        "- 请注意，批大小比以前小，因为我们要在这里执行后向传播，这更消耗内存"
    ])

    print(f"Writing {target_path}...")
    with open(target_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Done.")

if __name__ == "__main__":
    translate_notebook()
