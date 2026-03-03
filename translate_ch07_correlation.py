# -*- coding: utf-8 -*-
import json
import os

def translate_notebook(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    translations = {
        "Supplementary code for the <a href=\"http://mng.bz/orYv\">Build a Large Language Model From Scratch</a> book by <a href=\"https://sebastianraschka.com\">Sebastian Raschka</a><br>\n": "《<a href=\"http://mng.bz/orYv\">从头开始构建大型语言模型</a>》一书的补充代码，作者 <a href=\"https://sebastianraschka.com\">Sebastian Raschka</a><br>\n",
        "<br>Code repository: <a href=\"https://github.com/rasbt/LLMs-from-scratch\">https://github.com/rasbt/LLMs-from-scratch</a>\n": "<br>代码仓库：<a href=\"https://github.com/rasbt/LLMs-from-scratch\">https://github.com/rasbt/LLMs-from-scratch</a>\n",
        "# Score Correlation Analysis": "# 分数相关性分析",
        "- This notebook analyses the correlation between the different evaluation method scores": "- 本笔记本分析了不同评估方法分数之间的相关性",
        "### Correlation Coefficients": "### 相关系数",
        "- For comparison, below are the correlation coefficients from the Prometheus 2 paper by Kim et al. 2024 ([https://arxiv.org/abs/2405.01535](https://arxiv.org/abs/2405.01535)), which are all in the same ballpark as the ones reported for Llama 3 above\n": "- 为了比较，以下是 Kim 等人 2024 年 Prometheus 2 论文 ([https://arxiv.org/abs/2405.01535](https://arxiv.org/abs/2405.01535)) 中的相关系数，这些系数与上面报告的 Llama 3 的系数大致相同\n",
        "- Note that Prometheus 2 is a model specifically finetuned for LLM rating and evaluation ": "- 请注意，Prometheus 2 是专门为 LLM 评分和评估而微调的模型 "
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
        '/Users/richard/Git/LLMs-from-scratch/ch07/03_model-evaluation/scores/correlation-analysis.ipynb',
        '/Users/richard/Git/LLMs-from-scratch/ch07/03_model-evaluation/scores/correlation-analysis_zh.ipynb'
    )
