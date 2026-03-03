# -*- coding: utf-8 -*-
import json
import os

def translate_notebook():
    source_path = 'ch05/02_alternative_weight_loading/weight-loading-pytorch.ipynb'
    target_path = 'ch05/02_alternative_weight_loading/weight-loading-pytorch_zh.ipynb'
    
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
    set_source(36, ["## 从 PyTorch state dicts 加载权重的替代方法"])
    
    # Cell 43
    set_source(43, [
        "- 在主章节中，我们直接从 OpenAI 加载了 GPT 模型权重\n",
        "- 此笔记本提供替代的权重加载代码，用于从 PyTorch state dict 文件加载模型权重，这些文件是我从原始 TensorFlow 文件创建的，并上传到了 [Hugging Face Model Hub](https://huggingface.co/docs/hub/en/models-the-hub) 的 [https://huggingface.co/rasbt/gpt2-from-scratch-pytorch](https://huggingface.co/rasbt/gpt2-from-scratch-pytorch)\n",
        "- 这在概念上与通过第 5 章中描述的 state-dict 方法加载 PyTorch 模型的权重相同："
    ])
    
    # Cell 59
    set_source(59, ["### 选择模型"])
    
    # Cell 115
    set_source(115, ["### 下载文件"])
    
    # Cell 164
    set_source(164, ["### 加载权重"])
    
    # Cell 193
    set_source(193, ["### 生成文本"])
    
    # Cell 239
    set_source(239, ["## 替代的 safetensors 文件"])
    
    # Cell 246
    set_source(246, [
        "- 此外，[https://huggingface.co/rasbt/gpt2-from-scratch-pytorch](https://huggingface.co/rasbt/gpt2-from-scratch-pytorch) 存储库包含 state dicts 的所谓 `.safetensors` 版本\n",
        "- `.safetensors` 文件的吸引力在于其安全设计，因为它们只存储张量数据，避免了在加载过程中执行潜在的恶意代码\n",
        "- 在较新版本的 PyTorch（例如 2.0 及更高版本）中，可以将 `weights_only=True` 参数与 `torch.load` 一起使用（例如 `torch.load(\"model_state_dict.pth\", weights_only=True)`）以通过跳过代码执行并仅加载权重来提高安全性（这在 PyTorch 2.6 及更高版本中默认启用）；所以在这种情况下，从 state dict 文件加载权重应该不再是问题了"
    ])
    
    # Cell 250
    set_source(250, ["- 但是，下面的代码块简要展示了如何从这些 `.safetensor` 文件加载模型"])

    print(f"Writing {target_path}...")
    with open(target_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Done.")

if __name__ == "__main__":
    translate_notebook()
