
# -*- coding: utf-8 -*-
import json
import os

def translate_notebook(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    translations = {
        "## Exercise A.1": "## 练习 A.1",
        "The [Python Setup Tips](../../setup/01_optional-python-setup-preferences/README.md) document in this repository contains additional recommendations and tips to set up your Python environment.\n": "本仓库中的 [Python 设置提示](../../setup/01_optional-python-setup-preferences/README.md) 文档包含设置 Python 环境的其他建议和提示。\n",
        "## Exercise A.2": "## 练习 A.2",
        "The [Installing Libraries Used In This Book document](../../setup/02_installing-python-libraries/README.md) and [directory](../../setup/02_installing-python-libraries/) contains utilities to check whether your environment is set up correctly.": "[本书中使用的安装库文档](../../setup/02_installing-python-libraries/README.md) 和 [目录](../../setup/02_installing-python-libraries/) 包含检查您的环境是否设置正确的实用程序。",
        "## Exercise A.3": "## 练习 A.3",
        "## Exercise A.4": "## 练习 A.4"
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
    translate_notebook('appendix-A/01_main-chapter-code/exercise-solutions.ipynb', 'appendix-A/01_main-chapter-code/exercise-solutions_zh.ipynb')
