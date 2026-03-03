# -*- coding: utf-8 -*-
import json

# Input and output file paths
input_file = "ch05/13_olmo3/standalone-olmo3-plus-kv-cache.ipynb"
output_file = "ch05/13_olmo3/standalone-olmo3-plus-kv-cache_zh.ipynb"

# Translation mapping
translations = {
    "Supplementary code for the <a href=\"http://mng.bz/orYv\">Build a Large Language Model From Scratch</a> book by <a href=\"https://sebastianraschka.com\">Sebastian Raschka</a><br>": "Sebastian Raschka 的 <a href=\"http://mng.bz/orYv\">Build a Large Language Model From Scratch</a> 书籍的补充代码<br>",
    "<br>Code repository: <a href=\"https://github.com/rasbt/LLMs-from-scratch\">https://github.com/rasbt/LLMs-from-scratch</a>": "<br>代码仓库: <a href=\"https://github.com/rasbt/LLMs-from-scratch\">https://github.com/rasbt/LLMs-from-scratch</a>",
    "# Olmo 3 From Scratch (A Standalone Notebook)": "# 从头开始实现 Olmo 3 (独立笔记本)",
    "- This notebook is purposefully minimal and focuses on the code to re-implement Olmo 3 7B and 32 models from Allen AI in pure PyTorch without relying on other external LLM libraries; Olmo 3 is interesting because it is currently the leading fully open-source model": "- 本笔记本特意保持极简，专注于在纯 PyTorch 中重新实现 Allen AI 的 Olmo 3 7B 和 32 模型代码，而不依赖其他外部 LLM 库；Olmo 3 很有趣，因为它是目前领先的完全开源模型",
    "- For more information, see the official [Olmo 3 announcement](https://allenai.org/blog/olmo3) and model cards:": "- 有关更多信息，请参阅官方 [Olmo 3 公告](https://allenai.org/blog/olmo3) 和模型卡：",
    "- Note that there are also 32B versions, which are not listed above for brevity; you can find a complete list [here](https://huggingface.co/collections/allenai/olmo-3-post-training)": "- 请注意，还有 32B 版本，为简洁起见未在上面列出；您可以在 [这里](https://huggingface.co/collections/allenai/olmo-3-post-training) 找到完整列表",
    "- Below is a side-by-side comparison with Qwen3 8B as a reference model; if you are interested in the Qwen3 0.6B standalone notebook, you can find it [here](../11_qwen3)": "- 下面是与 Qwen3 8B 作为参考模型的并排比较；如果您对 Qwen3 0.6B 独立笔记本感兴趣，可以在 [这里](../11_qwen3) 找到它",
    "- About the code:": "- 关于代码：",
    "  - all code is my own code, mapping the Olmo 3 architecture onto the model code implemented in my [Build A Large Language Model (From Scratch)](http://mng.bz/orYv) book; the code is released under a permissive open-source Apache 2.0 license (see [LICENSE.txt](https://github.com/rasbt/LLMs-from-scratch/blob/main/LICENSE.txt))": "  - 所有代码均为我编写，将 Olmo 3 架构映射到我的 [Build A Large Language Model (From Scratch)](http://mng.bz/orYv) 书中实现与模型代码上；代码在宽松的开源 Apache 2.0 许可下发布（请参阅 [LICENSE.txt](https://github.com/rasbt/LLMs-from-scratch/blob/main/LICENSE.txt)）",
    "- Note that there are three model types, and each of the four model types comes in a 7B and 32B size:": "- 请注意，共有三种模型类型，每种模型类型都有 7B 和 32B 尺寸：",
    "# Select which model to use": "# 选择要使用的模型",
    "- In addition to the checkpoints listed above, you can also use the intermediate checkpoints listed [here](https://huggingface.co/collections/allenai/olmo-3-post-training); since they all have the same architecture, they are all compatible with this notebook": "- 除了上面列出的检查点之外，您还可以使用 [这里](https://huggingface.co/collections/allenai/olmo-3-post-training) 列出的中间检查点；由于它们都具有相同的架构，因此它们都与此笔记本兼容",
    "# 1. Architecture code": "# 1. 架构代码",
    "# 2. Initialize model": "# 2. 初始化模型",
    "- A quick check that the forward pass works before continuing:": "- 在继续之前快速检查前向传递是否工作：",
    "# 3. Load pretrained weights": "# 3. 加载预训练权重",
    "# 4. Load tokenizer": "# 4. 加载分词器",
    "# 5. Generate text": "# 5. 生成文本",
    "# What's next?": "# 下一步是什么？",
    "- For those interested in a comprehensive guide on building a large language model from scratch and gaining a deeper understanding of its mechanics, you might like my [Build a Large Language Model (From Scratch)](http://mng.bz/orYv)": "- 对于那些有兴趣从头开始构建大型语言模型并深入了解其机制的综合指南的人，您可能会喜欢我的 [Build a Large Language Model (From Scratch)](http://mng.bz/orYv)",
}

def translate_notebook(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            new_source = []
            for line in cell['source']:
                translated_line = line
                for eng, chi in translations.items():
                    if eng in translated_line:
                        translated_line = translated_line.replace(eng, chi)
                new_source.append(translated_line)
            cell['source'] = new_source

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

if __name__ == "__main__":
    translate_notebook(input_file, output_file)
    print(f"Translated {input_file} to {output_file}")
