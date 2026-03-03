# -*- coding: utf-8 -*-
import json
import os

def translate_notebook(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    translations = {
        "# Llama 3.2 From Scratch (A Standalone Notebook)": "# 从零开始实现 Llama 3.2 (独立笔记本)",
        "- This notebook is purposefully minimal and focuses on the code to implement the Llama 3.2 1B and 3B LLMs": "- 本笔记本特意保持极简，专注于实现 Llama 3.2 1B 和 3B LLM 的代码",
        "- For a step-by-step guide that explains the individual components and the relationship between GPT, Llama 2, and Llama 3, please see the following companion notebooks:": "- 如果需要解释各个组件以及 GPT、Llama 2 和 Llama 3 之间关系的逐步指南，请参阅以下配套笔记本：",
        "  - [Converting a From-Scratch GPT Architecture to Llama 2](converting-gpt-to-llama2.ipynb)": "  - [将从零开始的 GPT 架构转换为 Llama 2](converting-gpt-to-llama2.ipynb)",
        "  - [Converting Llama 2 to Llama 3.2 From Scratch](converting-llama2-to-llama3.ipynb)": "  - [从零开始将 Llama 2 转换为 Llama 3.2](converting-llama2-to-llama3.ipynb)",
        "- About the code:": "- 关于代码：",
        "  - all code is my own code, mapping the Llama 3 architecture onto the model code implemented in my [Build A Large Language Model (From Scratch)](http://mng.bz/orYv) book; the code is released under a permissive open-source Apache 2.0 license (see [LICENSE.txt](https://github.com/rasbt/LLMs-from-scratch/blob/main/LICENSE.txt))": "  - 所有代码均为我自己编写，将 Llama 3 架构映射到我在 [Build A Large Language Model (From Scratch)](http://mng.bz/orYv) 一书中实现的代码上；代码在宽松的开源 Apache 2.0 许可下发布（请参阅 [LICENSE.txt](https://github.com/rasbt/LLMs-from-scratch/blob/main/LICENSE.txt)）",
        "  - the tokenizer code is inspired by the original [Llama 3 tokenizer code](https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py), which Meta AI used to extend the Tiktoken GPT-4 tokenizer": "  - 分词器代码灵感来自原始的 [Llama 3 分词器代码](https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py)，Meta AI 使用该代码扩展了 Tiktoken GPT-4 分词器",
        "  - the RoPE rescaling section is inspired by the [_compute_llama3_parameters function](https://github.com/huggingface/transformers/blob/5c1027bf09717f664b579e01cbb8ec3ef5aeb140/src/transformers/modeling_rope_utils.py#L329-L347) in the `transformers` library": "  - RoPE 重缩放部分灵感来自 `transformers` 库中的 [_compute_llama3_parameters 函数](https://github.com/huggingface/transformers/blob/5c1027bf09717f664b579e01cbb8ec3ef5aeb140/src/transformers/modeling_rope_utils.py#L329-L347)",
        "# 1. Architecture code": "# 1. 架构代码",
        "# 2. Initialize model": "# 2. 初始化模型",
        "- The remainder of this notebook uses the Llama 3.2 1B model; to use the 3B model variant, just uncomment the second configuration file in the following code cell": "- 本笔记本的其余部分使用 Llama 3.2 1B 模型；要使用 3B 模型变体，只需在下面的代码单元格中取消注释第二个配置文件",
        "# 3. Load tokenizer": "# 3. 加载分词器",
        "- Please note that Meta AI requires that you accept the Llama 3.2 licensing terms before you can download the files; to do this, you have to create a Hugging Face Hub account and visit the [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) repository to accept the terms": "- 请注意，Meta AI 要求您在下载文件之前接受 Llama 3.2 许可条款；为此，您必须创建一个 Hugging Face Hub 帐户并访问 [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) 存储库以接受条款",
        "- Next, you will need to create an access token; to generate an access token with READ permissions, click on the profile picture in the upper right and click on \"Settings\"": "- 接下来，您需要创建一个访问令牌；要生成具有读取权限的访问令牌，请单击右上角的个人资料图片，然后单击“Settings”",
        "- Then, create and copy the access token so you can copy & paste it into the next code cell": "- 然后，创建并复制访问令牌，以便您可以将其复制并粘贴到下一个代码单元格中",
        "# 4. Load pretrained weights": "# 4. 加载预训练权重",
        "# 5. Generate text": "# 5. 生成文本",
        "# What's next?": "# 下一步是什么？",
        "- The notebook was kept purposefully minimal; if you are interested in additional explanation about the individual components, check out the following two companion notebooks:": "- 本笔记本特意保持极简；如果您对各个组件的额外解释感兴趣，请查看以下两个配套笔记本：",
        "  1. [Converting a From-Scratch GPT Architecture to Llama 2](converting-gpt-to-llama2.ipynb)": "  1. [将从零开始的 GPT 架构转换为 Llama 2](converting-gpt-to-llama2.ipynb)",
        "  2. [Converting Llama 2 to Llama 3.2 From Scratch](converting-llama2-to-llama3.ipynb)": "  2. [从零开始将 Llama 2 转换为 Llama 3.2](converting-llama2-to-llama3.ipynb)",
        "- For those interested in a comprehensive guide on building a large language model from scratch and gaining a deeper understanding of its mechanics, you might like my [Build a Large Language Model (From Scratch)](http://mng.bz/orYv)": "- 对于那些有兴趣了解从零开始构建大型语言模型的综合指南并深入了解其机制的人，您可能会喜欢我的 [Build a Large Language Model (From Scratch)](http://mng.bz/orYv)",
    }

    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            new_source = []
            for line in cell['source']:
                translated_line = line
                # Try exact match first (stripped)
                stripped_line = line.strip()
                if stripped_line in translations:
                     # Preserve indentation/newlines if possible, but simplest is to just replace content if it's a full line match
                     # However, my dictionary keys are exact strings found in the source lines (mostly).
                     # Some are parts of lines.
                     pass
                
                for eng, chi in translations.items():
                    if eng in line:
                        translated_line = line.replace(eng, chi)
                        break
                new_source.append(translated_line)
            cell['source'] = new_source

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

if __name__ == "__main__":
    print("Reading ch05/07_gpt_to_llama/standalone-llama32.ipynb...")
    translate_notebook('ch05/07_gpt_to_llama/standalone-llama32.ipynb', 'ch05/07_gpt_to_llama/standalone-llama32_zh.ipynb')
    print("Done.")
