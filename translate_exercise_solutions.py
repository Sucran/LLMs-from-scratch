import json

file_path = '/Users/richard/Git/LLMs-from-scratch/ch03/01_main-chapter-code/exercise-solutions.ipynb'
output_path = '/Users/richard/Git/LLMs-from-scratch/ch03/01_main-chapter-code/exercise-solutions_zh.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def set_cell(idx, text):
    if idx < len(nb['cells']):
        nb['cells'][idx]['source'] = [line + '\n' for line in text.split('\n')]
        if nb['cells'][idx]['source']:
            nb['cells'][idx]['source'][-1] = nb['cells'][idx]['source'][-1].rstrip('\n')

set_cell(0, '<table style="width:100%">\n<tr>\n<td style="vertical-align:middle; text-align:left;">\n<font size="2">\n<a href="http://mng.bz/orYv">Build a Large Language Model From Scratch</a> 书籍的补充代码，作者 <a href="https://sebastianraschka.com">Sebastian Raschka</a><br>\n<br>代码仓库：<a href="https://github.com/rasbt/LLMs-from-scratch">https://github.com/rasbt/LLMs-from-scratch</a>\n</font>\n</td>\n<td style="vertical-align:middle; text-align:left;">\n<a href="http://mng.bz/orYv"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/cover-small.webp" width="100px"></a>\n</td>\n</tr>\n</table>')
set_cell(1, '# 第 3 章练习解答')
set_cell(3, '## 练习 3.1')
set_cell(10, '## 练习 3.2')
set_cell(11, '如果我们想要输出维度为 2，就像早期的单头注意力一样，我们必须将投影维度 `d_out` 更改为 1：')
set_cell(14, '## 练习 3.3')
set_cell(16, '可选地，参数数量如下：')
set_cell(19, 'GPT-2 模型总共有 1.17 亿个参数，但正如我们所见，它的大部分参数并不在多头注意力模块本身。')

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
