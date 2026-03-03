
# -*- coding: utf-8 -*-
import json
import os

def translate_notebook(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    translations = {
        "# Appendix A: Introduction to PyTorch (Part 1)": "# 附录 A：PyTorch 简介 (第 1 部分)",
        "## A.1 What is PyTorch": "## A.1 什么是 PyTorch",
        "## A.2 Understanding tensors": "## A.2 理解张量",
        "### A.2.1 Scalars, vectors, matrices, and tensors": "### A.2.1 标量、向量、矩阵和张量",
        "### A.2.2 Tensor data types": "### A.2.2 张量数据类型",
        "### A.2.3 Common PyTorch tensor operations": "### A.2.3 常见的 PyTorch 张量操作",
        "## A.3 Seeing models as computation graphs": "## A.3 将模型视为计算图",
        "## A.4 Automatic differentiation made easy": "## A.4 自动微分变得简单",
        "## A.5 Implementing multilayer neural networks": "## A.5 实现多层神经网络",
        "## A.6 Setting up efficient data loaders": "## A.6 设置高效的数据加载器",
        "## A.7 A typical training loop": "## A.7 典型的训练循环",
        "## A.8 Saving and loading models": "## A.8 保存和加载模型",
        "## A.9 Optimizing training performance with GPUs": "## A.9 使用 GPU 优化训练性能",
        "### A.9.1 PyTorch computations on GPU devices": "### A.9.1 GPU 设备上的 PyTorch 计算",
        "See [code-part2.ipynb](code-part2.ipynb)": "参见 [code-part2.ipynb](code-part2.ipynb)",
        "### A.9.2 Single-GPU training": "### A.9.2 单 GPU 训练",
        "### A.9.3 Training with multiple GPUs": "### A.9.3 多 GPU 训练",
        "See [DDP-script.py](DDP-script.py)": "参见 [DDP-script.py](DDP-script.py)"
    }

    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            new_source = []
            for line in cell['source']:
                translated_line = translations.get(line.strip(), line)
                if line.strip() in translations:
                     new_source.append(translated_line + ('\n' if line.endswith('\n') else ''))
                else:
                    new_source.append(line)
            cell['source'] = new_source

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

if __name__ == "__main__":
    translate_notebook('appendix-A/01_main-chapter-code/code-part1.ipynb', 'appendix-A/01_main-chapter-code/code-part1_zh.ipynb')
