# -*- coding: utf-8 -*-
import json
import os

# Translation dictionary
trans_map = {
    "Supplementary code for the <a href=\"http://mng.bz/orYv\">Build a Large Language Model From Scratch</a> book by <a href=\"https://sebastianraschka.com\">Sebastian Raschka</a><br>": "本书 <a href=\"http://mng.bz/orYv\">Build a Large Language Model From Scratch</a> 的补充代码，作者 <a href=\"https://sebastianraschka.com\">Sebastian Raschka</a><br>",
    "Code repository: <a href=\"https://github.com/rasbt/LLMs-from-scratch\">https://github.com/rasbt/LLMs-from-scratch</a>": "代码仓库：<a href=\"https://github.com/rasbt/LLMs-from-scratch\">https://github.com/rasbt/LLMs-from-scratch</a>",
    "# Scikit-learn Logistic Regression Model": "# Scikit-learn 逻辑回归模型",
    "## Scikit-learn baseline": "## Scikit-learn 基线"
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
    input_file = "/Users/richard/Git/LLMs-from-scratch/ch06/03_bonus_imdb-classification/sklearn-baseline.ipynb"
    output_file = "/Users/richard/Git/LLMs-from-scratch/ch06/03_bonus_imdb-classification/sklearn-baseline_zh.ipynb"
    translate_notebook(input_file, output_file)
    print(f"Translated {input_file} to {output_file}")
