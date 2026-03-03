
# -*- coding: utf-8 -*-
import json
import os

def translate_notebook(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    translations = {
        "# Appendix A: Introduction to PyTorch (Part 2)": "# 附录 A：PyTorch 简介 (第 2 部分)",
        "## A.9 Optimizing training performance with GPUs": "## A.9 使用 GPU 优化训练性能",
        "### A.9.1 PyTorch computations on GPU devices": "### A.9.1 GPU 设备上的 PyTorch 计算",
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
    translate_notebook('appendix-A/01_main-chapter-code/code-part2.ipynb', 'appendix-A/01_main-chapter-code/code-part2_zh.ipynb')
