# -*- coding: utf-8 -*-
import json
import os

def translate_notebook(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    translations = {
        "Supplementary code for the <a href=\"http://mng.bz/orYv\">Build a Large Language Model From Scratch</a> book by <a href=\"https://sebastianraschka.com\">Sebastian Raschka</a><br>\n": "《<a href=\"http://mng.bz/orYv\">从头开始构建大型语言模型</a>》一书的补充代码，作者 <a href=\"https://sebastianraschka.com\">Sebastian Raschka</a><br>\n",
        "<br>Code repository: <a href=\"https://github.com/rasbt/LLMs-from-scratch\">https://github.com/rasbt/LLMs-from-scratch</a>\n": "<br>代码仓库：<a href=\"https://github.com/rasbt/LLMs-from-scratch\">https://github.com/rasbt/LLMs-from-scratch</a>\n",
        "# Load And Use Finetuned Model": "# 加载并使用微调后的模型",
        "This notebook contains minimal code to load the finetuned model that was instruction finetuned and saved in chapter 7 via [ch07.ipynb](ch07.ipynb).": "本笔记本包含用于加载在第 7 章中通过 [ch07.ipynb](ch07.ipynb) 进行指令微调并保存的微调模型的最小代码。"
    }

    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            for i, line in enumerate(cell['source']):
                if line in translations:
                    cell['source'][i] = translations[line]

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

if __name__ == "__main__":
    translate_notebook(
        '/Users/richard/Git/LLMs-from-scratch/ch07/01_main-chapter-code/load-finetuned-model.ipynb',
        '/Users/richard/Git/LLMs-from-scratch/ch07/01_main-chapter-code/load-finetuned-model_zh.ipynb'
    )
