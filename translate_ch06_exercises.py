# -*- coding: utf-8 -*-
import json
import os

# Translation dictionary
trans_map = {
    "Supplementary code for the <a href=\"http://mng.bz/orYv\">Build a Large Language Model From Scratch</a> book by <a href=\"https://sebastianraschka.com\">Sebastian Raschka</a><br>": "本书 <a href=\"http://mng.bz/orYv\">Build a Large Language Model From Scratch</a> 的补充代码，作者 <a href=\"https://sebastianraschka.com\">Sebastian Raschka</a><br>",
    "Code repository: <a href=\"https://github.com/rasbt/LLMs-from-scratch\">https://github.com/rasbt/LLMs-from-scratch</a>": "代码仓库：<a href=\"https://github.com/rasbt/LLMs-from-scratch\">https://github.com/rasbt/LLMs-from-scratch</a>",
    "# Chapter 6 Exercise solutions": "# 第 6 章 练习解答",
    "## Exercise 6.1: Increasing the context length": "## 练习 6.1：增加上下文长度",
    "We can pad the inputs to the maximum number of tokens the model supports by setting the max length to 1024:\n\n```python\nmax_length = 1024\n\ntrain_dataset = SpamDataset(base_path / \"train.csv\", max_length=max_length, tokenizer=tokenizer)\nval_dataset = SpamDataset(base_path / \"validation.csv\", max_length=max_length, tokenizer=tokenizer)\ntest_dataset = SpamDataset(base_path / \"test.csv\", max_length=max_length, tokenizer=tokenizer)\n```\n\nor, equivalently, we can define the `max_length` via:\n\n```python\nmax_length = model.pos_emb.weight.shape[0]\n```\n\nor\n\n```python\nmax_length = BASE_CONFIG[\"context_length\"]\n```": "我们可以通过将最大长度设置为 1024，将输入填充到模型支持的最大标记数：\n\n```python\nmax_length = 1024\n\ntrain_dataset = SpamDataset(base_path / \"train.csv\", max_length=max_length, tokenizer=tokenizer)\nval_dataset = SpamDataset(base_path / \"validation.csv\", max_length=max_length, tokenizer=tokenizer)\ntest_dataset = SpamDataset(base_path / \"test.csv\", max_length=max_length, tokenizer=tokenizer)\n```\n\n或者，等效地，我们可以通过以下方式定义 `max_length`：\n\n```python\nmax_length = model.pos_emb.weight.shape[0]\n```\n\n或者\n\n```python\nmax_length = BASE_CONFIG[\"context_length\"]\n```",
    "For convenience, you can run this experiment via\n\n```bash\npython additional-experiments.py --context_length \"model_context_length\"\n```\n\nusing the code in the [../02_bonus_additional-experiments](../02_bonus_additional-experiments) folder, which results in a substantially worse test accuracy of 78.33% (versus the 95.67% in the main chapter).": "为了方便起见，您可以通过以下方式运行此实验\n\n```bash\npython additional-experiments.py --context_length \"model_context_length\"\n```\n\n使用 [../02_bonus_additional-experiments](../02_bonus_additional-experiments) 文件夹中的代码，这将导致测试准确率显着降低，为 78.33%（而主要章节中为 95.67%）。",
    "## Exercise 6.2: Finetuning the whole model": "## 练习 6.2：微调整个模型",
    "Instead of finetuning just the final transformer block, we can finetune the entire model by removing the following lines from the code:\n\n```python\nfor param in model.parameters():\n    param.requires_grad = False\n```\n\nFor convenience, you can run this experiment via\n\n```bash\npython additional-experiments.py --trainable_layers all\n```\n\nusing the code in the [../02_bonus_additional-experiments](../02_bonus_additional-experiments) folder, which results in a 1% improved test accuracy of 96.67% (versus the 95.67% in the main chapter).": "我们可以微调整个模型，而不是仅微调最后一个 transformer 块，方法是从代码中删除以下行：\n\n```python\nfor param in model.parameters():\n    param.requires_grad = False\n```\n\n为了方便起见，您可以通过以下方式运行此实验\n\n```bash\npython additional-experiments.py --trainable_layers all\n```\n\n使用 [../02_bonus_additional-experiments](../02_bonus_additional-experiments) 文件夹中的代码，这将导致测试准确率提高 1%，达到 96.67%（而主要章节中为 95.67%）。",
    "## Exercise 6.3: Finetuning the first versus last token ": "## 练习 6.3：微调第一个与最后一个标记 ",
    "Rather than finetuning the last output token, we can finetune the first output token by changing \n\n```python\nmodel(input_batch)[:, -1, :]\n```\n\nto\n\n```python\nmodel(input_batch)[:, 0, :]\n```\n\neverywhere in the code.\n\nFor convenience, you can run this experiment via\n\n```\npython additional-experiments.py --trainable_token first\n```\n\nusing the code in the [../02_bonus_additional-experiments](../02_bonus_additional-experiments) folder, which results in a substantially worse test accuracy of 75.00% (versus the 95.67% in the main chapter).": "我们可以微调第一个输出标记，而不是微调最后一个输出标记，方法是将代码中所有出现的\n\n```python\nmodel(input_batch)[:, -1, :]\n```\n\n更改为\n\n```python\nmodel(input_batch)[:, 0, :]\n```\n\n。\n\n为了方便起见，您可以通过以下方式运行此实验\n\n```\npython additional-experiments.py --trainable_token first\n```\n\n使用 [../02_bonus_additional-experiments](../02_bonus_additional-experiments) 文件夹中的代码，这将导致测试准确率显着降低，为 75.00%（而主要章节中为 95.67%）。"
}

def translate_notebook(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            new_source = []
            for line in cell['source']:
                translated_line = line
                # Try exact match first (stripping newline for matching, but keeping it for output)
                line_stripped = line.strip()
                if line_stripped in trans_map:
                    translated_line = trans_map[line_stripped]
                    if line.endswith('\n'):
                        translated_line += '\n'
                else:
                     # If no exact match, try to match parts if needed, or check if it's a multiline block in dict
                     # Join lines to check for multiline matches in the dictionary
                     pass
                new_source.append(translated_line)
            
            # Re-check for full cell content match (handling multiline strings in the dict)
            full_source = "".join(cell['source'])
            if full_source in trans_map:
                cell['source'] = [trans_map[full_source]]
            else:
                # Iterate again to apply single line translations if full source didn't match
                final_source = []
                for line in cell['source']:
                    line_content = line.rstrip('\n')
                    if line_content in trans_map:
                        trans_content = trans_map[line_content]
                        if line.endswith('\n'):
                            trans_content += '\n'
                        final_source.append(trans_content)
                    else:
                        final_source.append(line)
                cell['source'] = final_source

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

if __name__ == "__main__":
    input_file = "/Users/richard/Git/LLMs-from-scratch/ch06/01_main-chapter-code/exercise-solutions.ipynb"
    output_file = "/Users/richard/Git/LLMs-from-scratch/ch06/01_main-chapter-code/exercise-solutions_zh.ipynb"
    translate_notebook(input_file, output_file)
    print(f"Translated {input_file} to {output_file}")
