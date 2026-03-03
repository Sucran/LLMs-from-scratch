import json

file_path = '/Users/richard/Git/LLMs-from-scratch/ch03/01_main-chapter-code/multihead-attention.ipynb'
output_path = '/Users/richard/Git/LLMs-from-scratch/ch03/01_main-chapter-code/multihead-attention_zh.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def set_cell(idx, text):
    if idx < len(nb['cells']):
        nb['cells'][idx]['source'] = [line + '\n' for line in text.split('\n')]
        if nb['cells'][idx]['source']:
            nb['cells'][idx]['source'][-1] = nb['cells'][idx]['source'][-1].rstrip('\n')

set_cell(0, '<table style="width:100%">\n<tr>\n<td style="vertical-align:middle; text-align:left;">\n<font size="2">\n<a href="http://mng.bz/orYv">Build a Large Language Model From Scratch</a> 书籍的补充代码，作者 <a href="https://sebastianraschka.com">Sebastian Raschka</a><br>\n<br>代码仓库：<a href="https://github.com/rasbt/LLMs-from-scratch">https://github.com/rasbt/LLMs-from-scratch</a>\n</font>\n</td>\n<td style="vertical-align:middle; text-align:left;">\n<a href="http://mng.bz/orYv"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/cover-small.webp" width="100px"></a>\n</td>\n</tr>\n</table>')
set_cell(1, '# 多头注意力加数据加载')
set_cell(3, '完整的章节代码位于 [ch03.ipynb](./ch03.ipynb)。\n\n本笔记本包含主要内容，即多头注意力实现（加上第 2 章的数据加载管道）')
set_cell(4, '## 第 2 章的数据加载器')
set_cell(8, '# 第 3 章的多头注意力')
set_cell(9, '## 变体 A：简单实现')
set_cell(12, '## 变体 B：替代实现')

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
