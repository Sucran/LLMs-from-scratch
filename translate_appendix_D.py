
# -*- coding: utf-8 -*-
import json
import os

def translate_notebook(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    translations = {
        "# Appendix D: Adding Bells and Whistles to the Training Loop": "# 附录 D：为训练循环添加花哨的功能",
        "## D.1 Learning rate warmup": "## D.1 学习率预热",
        "## D.2 Cosine decay": "## D.2 余弦衰减",
        "## D.3 Gradient clipping": "## D.3 梯度裁剪",
        "## D.4 The modified training function": "## D.4 修改后的训练函数"
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
    translate_notebook('appendix-D/01_main-chapter-code/appendix-D.ipynb', 'appendix-D/01_main-chapter-code/appendix-D_zh.ipynb')
