# -*- coding: utf-8 -*-
import json
import os

def translate_notebook():
    source_path = 'ch05/02_alternative_weight_loading/weight-loading-hf-safetensors.ipynb'
    target_path = 'ch05/02_alternative_weight_loading/weight-loading-hf-safetensors_zh.ipynb'
    
    print(f"Reading {source_path}...")
    with open(source_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb['cells']
    
    def set_source(idx, text_list):
        if 0 <= idx < len(cells):
            if isinstance(text_list, str):
                text_list = [text_list]
            cells[idx]['source'] = text_list

    # Cell 28
    set_source(28, ["# 第 5 章 奖励代码"])
    
    # Cell 36
    set_source(36, ["## 使用 `safetensors` 从 Hugging Face Model Hub 加载权重的替代方法"])
    
    # Cell 44
    set_source(44, [
        "- 在主章节中，我们直接从 OpenAI 加载了 GPT 模型权重\n",
        "- 此笔记本提供替代的权重加载代码，使用 `.safetensors` 文件从 [Hugging Face Model Hub](https://huggingface.co/docs/hub/en/models-the-hub) 加载模型权重\n",
        "- 这在概念上与通过第 5 章中描述的 state-dict 方法加载 PyTorch 模型的权重相同："
    ])

    print(f"Writing {target_path}...")
    with open(target_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Done.")

if __name__ == "__main__":
    translate_notebook()
