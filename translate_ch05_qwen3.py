# -*- coding: utf-8 -*-
import json
import os

def translate_notebook(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    translations = {
        "# Qwen3 From Scratch (A Standalone Notebook)": "# 从零开始实现 Qwen3 (独立笔记本)",
        "- This notebook is purposefully minimal and focuses on the code to implement Qwen3 0.6B, 1.7B, 4B, 8B, and 32B; for more information about this model, please see the original blog post and technical report:": "- 本笔记本特意保持极简，专注于实现 Qwen3 0.6B、1.7B、4B、8B 和 32B 的代码；有关该模型的更多信息，请参阅原始博客文章和技术报告：",
        "  - [Qwen3: Think Deeper, Act Faster](https://qwenlm.github.io/blog/qwen3/)": "  - [Qwen3: 思考更深，行动更快](https://qwenlm.github.io/blog/qwen3/)",
        "  - [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)": "  - [Qwen3 技术报告](https://arxiv.org/abs/2505.09388)",
        "- Many architectural components in Qwen3 are similar to Llama 3; for a step-by-step guide that explains the individual components and the relationship between GPT and the components used here, you may like the GPT-to-Llama conversion notebooks:": "- Qwen3 中的许多架构组件与 Llama 3 相似；如果需要解释各个组件以及 GPT 与此处使用的组件之间关系的逐步指南，您可能会喜欢 GPT 到 Llama 的转换笔记本：",
        "  - [Converting a From-Scratch GPT Architecture to Llama 2](../07_gpt_to_llama/converting-gpt-to-llama2.ipynb)": "  - [将从零开始的 GPT 架构转换为 Llama 2](../07_gpt_to_llama/converting-gpt-to-llama2.ipynb)",
        "  - [Converting Llama 2 to Llama 3.2 From Scratch](../07_gpt_to_llama/converting-llama2-to-llama3.ipynb)": "  - [从零开始将 Llama 2 转换为 Llama 3.2](../07_gpt_to_llama/converting-llama2-to-llama3.ipynb)",
        "- About the code:": "- 关于代码：",
        "  - all code is my own code, mapping the Qwen3 architecture onto the model code implemented in my [Build A Large Language Model (From Scratch)](http://mng.bz/orYv) book; the code is released under a permissive open-source Apache 2.0 license (see [LICENSE.txt](https://github.com/rasbt/LLMs-from-scratch/blob/main/LICENSE.txt))": "  - 所有代码均为我自己编写，将 Qwen3 架构映射到我在 [Build A Large Language Model (From Scratch)](http://mng.bz/orYv) 一书中实现的代码上；代码在宽松的开源 Apache 2.0 许可下发布（请参阅 [LICENSE.txt](https://github.com/rasbt/LLMs-from-scratch/blob/main/LICENSE.txt)）",
        "- Note that there are two models, the \"base\" and the \"hybrid\" model, and the hybrid model can be used as either a reasoning or a regular instruction-following model:": "- 请注意，有两种模型，“base”和“hybrid”模型，hybrid 模型既可以用作推理模型，也可以用作常规指令跟随模型：",
        "- In short, the model types are as follows:": "- 简而言之，模型类型如下：",
        "  - `base`: the pretrained base model; note that the Qwen3 pretraining contained some reasoning data (chain-of-thought data), so the model sometimes emits reasoning traces even though it didn't undergo the reasoning training (reinforcement learning) stages": "  - `base`：预训练的基础模型；请注意，Qwen3 预训练包含一些推理数据（思维链数据），因此即使没有经过推理训练（强化学习）阶段，该模型有时也会发出推理轨迹",
        "  - `hybrid`": "  - `hybrid`",
        "    - `reasoning`: emits long reasoning traces inside `<think></think>` tags": "    - `reasoning`：在 `<think></think>` 标签内发出长推理轨迹",
        "    - `instruct`: the same as above, but long reasoning traces can be suppressed by manually adding empty `<think></think>` (this is done by the tokenizer); this way, the model acts like a regular instruction-following model": "    - `instruct`：同上，但可以通过手动添加空的 `<think></think>` 来抑制长推理轨迹（这由分词器完成）；这样，该模型的行为就像常规指令跟随模型一样",
        "# 1. Architecture code": "# 1. 架构代码",
        "# 2. Initialize model": "# 2. 初始化模型",
        "- A quick check that the forward pass works before continuing:": "- 在继续之前快速检查前向传递是否工作：",
        "# 3. Load pretrained weights": "# 3. 加载预训练权重",
        "# 4. Load tokenizer": "# 4. 加载分词器",
        "# 4. Generate text": "# 4. 生成文本",
        "# What's next?": "# 下一步是什么？",
        "- Check out the [README.md](./README.md), to use this model via the `llms_from_scratch` package": "- 查看 [README.md](./README.md)，通过 `llms_from_scratch` 包使用此模型",
        "- For those interested in a comprehensive guide on building a large language model from scratch and gaining a deeper understanding of its mechanics, you might like my [Build a Large Language Model (From Scratch)](http://mng.bz/orYv)": "- 对于那些有兴趣了解从零开始构建大型语言模型的综合指南并深入了解其机制的人，您可能会喜欢我的 [Build a Large Language Model (From Scratch)](http://mng.bz/orYv)",
    }

    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            new_source = []
            for line in cell['source']:
                translated_line = line
                for eng, chi in translations.items():
                    if eng in line:
                        translated_line = line.replace(eng, chi)
                        break
                new_source.append(translated_line)
            cell['source'] = new_source

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

if __name__ == "__main__":
    print("Reading ch05/11_qwen3/standalone-qwen3.ipynb...")
    translate_notebook('ch05/11_qwen3/standalone-qwen3.ipynb', 'ch05/11_qwen3/standalone-qwen3_zh.ipynb')
    print("Done.")
