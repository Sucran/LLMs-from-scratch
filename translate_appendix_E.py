
# -*- coding: utf-8 -*-
import json
import os

def translate_notebook(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    translations = {
        "# Appendix E: Parameter-efficient Finetuning with LoRA": "# 附录 E：使用 LoRA 进行参数高效微调",
        "## E.1 Introduction to LoRA": "## E.1 LoRA 简介",
        "## E.2 Preparing the dataset": "## E.2 准备数据集",
        "## E.3 Initializing the model": "## E.3 初始化模型",
        "## E.4 Parameter-efficient finetuning with LoRA": "## E.4 使用 LoRA 进行参数高效微调"
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
    translate_notebook('appendix-E/01_main-chapter-code/appendix-E.ipynb', 'appendix-E/01_main-chapter-code/appendix-E_zh.ipynb')
