# -*- coding: utf-8 -*-
import json

# Input and output file paths
input_file = "ch05/11_qwen3/standalone-qwen3-moe-plus-kvcache.ipynb"
output_file = "ch05/11_qwen3/standalone-qwen3-moe-plus-kvcache_zh.ipynb"

# Translation mapping
translations = {
    "Supplementary code for the <a href=\"http://mng.bz/orYv\">Build a Large Language Model From Scratch</a> book by <a href=\"https://sebastianraschka.com\">Sebastian Raschka</a><br>": "Sebastian Raschka 的 <a href=\"http://mng.bz/orYv\">Build a Large Language Model From Scratch</a> 书籍的补充代码<br>",
    "<br>Code repository: <a href=\"https://github.com/rasbt/LLMs-from-scratch\">https://github.com/rasbt/LLMs-from-scratch</a>": "<br>代码仓库: <a href=\"https://github.com/rasbt/LLMs-from-scratch\">https://github.com/rasbt/LLMs-from-scratch</a>",
    "# Qwen3 Mixture-of-Experts From Scratch (A Standalone Notebook)": "# 从头开始实现 Qwen3 混合专家模型 (独立笔记本)",
    "- This notebook is purposefully minimal and focuses on the code to implement Qwen3-30B-A3B model (with support for **Coder**, **Instruct** and **Thinking** variants); for more information about this model, please see the original blog post, technical report, and model hub pages:": "- 本笔记本特意保持极简，专注于实现 Qwen3-30B-A3B 模型（支持 **Coder**、**Instruct** 和 **Thinking** 变体）的代码；有关此模型的更多信息，请参阅原始博客文章、技术报告和模型中心页面：",
    "- Many architectural components in Qwen3 are similar to Llama 3; for a step-by-step guide that explains the individual components and the relationship between GPT and the components used here, you may like the GPT-to-Llama conversion notebooks:": "- Qwen3 中的许多架构组件与 Llama 3 相似；如果需要逐步解释各个组件以及 GPT 与此处使用的组件之间关系的指南，您可能会喜欢 GPT 到 Llama 的转换笔记本：",
    "  - [Converting a From-Scratch GPT Architecture to Llama 2](../07_gpt_to_llama/converting-gpt-to-llama2.ipynb)": "  - [将从头开始的 GPT 架构转换为 Llama 2](../07_gpt_to_llama/converting-gpt-to-llama2_zh.ipynb)",
    "  - [Converting Llama 2 to Llama 3.2 From Scratch](../07_gpt_to_llama/converting-llama2-to-llama3.ipynb)": "  - [从头开始将 Llama 2 转换为 Llama 3.2](../07_gpt_to_llama/converting-llama2-to-llama3_zh.ipynb)",
    "**By default, this notebook runs Qwen3-Coder-30B-A3B-Instruct (aka Qwen3 Coder Flash) and requires 80 GB of VRAM (e.g., a single A100 or H100)**": "**默认情况下，此笔记本运行 Qwen3-Coder-30B-A3B-Instruct（又名 Qwen3 Coder Flash），需要 80 GB 的 VRAM（例如，单个 A100 或 H100）**",
    "- About the code:": "- 关于代码：",
    "  - all code is my own code, mapping the Qwen3 architecture onto the model code implemented in my [Build A Large Language Model (From Scratch)](http://mng.bz/orYv) book; the code is released under a permissive open-source Apache 2.0 license (see [LICENSE.txt](https://github.com/rasbt/LLMs-from-scratch/blob/main/LICENSE.txt))": "  - 所有代码均为我编写，将 Qwen3 架构映射到我的 [Build A Large Language Model (From Scratch)](http://mng.bz/orYv) 书中实现与模型代码上；代码在宽松的开源 Apache 2.0 许可下发布（请参阅 [LICENSE.txt](https://github.com/rasbt/LLMs-from-scratch/blob/main/LICENSE.txt)）",
    "# 1. Architecture code": "# 1. 架构代码",
    "# 2. Initialize model": "# 2. 初始化模型",
    "- A quick check that the forward pass works before continuing (nan values are ok for now since we are using a \"meta\" device upon instantiation to save memory):": "- 在继续之前快速检查前向传递是否工作（现在出现 nan 值是可以的，因为我们在实例化时使用了 \"meta\" 设备以节省内存）：",
    "- Don't be concerned; the model runs fine on an A100 card with 80 GB RAM due to offloading some layers to CPU RAM": "- 不用担心；由于将某些层卸载到 CPU RAM，该模型在具有 80 GB RAM 的 A100 卡上运行良好",
    "# 3. Load pretrained weights": "# 3. 加载预训练权重",
    "# 4. Load tokenizer": "# 4. 加载分词器",
    "# 5. Generate text": "# 5. 生成文本",
    "# What's next?": "# 下一步是什么？",
    "- Check out the [README.md](./README.md), to use this model via the `llms_from_scratch` package": "- 查看 [README.md](./README_zh.md)，通过 `llms_from_scratch` 包使用此模型",
    "- For those interested in a comprehensive guide on building a large language model from scratch and gaining a deeper understanding of its mechanics, you might like my [Build a Large Language Model (From Scratch)](http://mng.bz/orYv)": "- 对于那些有兴趣从头开始构建大型语言模型并深入了解其机制的综合指南的人，您可能会喜欢我的 [Build a Large Language Model (From Scratch)](http://mng.bz/orYv)",
    "README.md": "README_zh.md"
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
