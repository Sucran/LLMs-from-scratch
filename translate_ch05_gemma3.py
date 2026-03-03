# -*- coding: utf-8 -*-
import json

# Input and output file paths
input_file = "ch05/12_gemma3/standalone-gemma3.ipynb"
output_file = "ch05/12_gemma3/standalone-gemma3_zh.ipynb"

# Translation mapping
translations = {
    "Supplementary code for the <a href=\"http://mng.bz/orYv\">Build a Large Language Model From Scratch</a> book by <a href=\"https://sebastianraschka.com\">Sebastian Raschka</a><br>": "Sebastian Raschka 的 <a href=\"http://mng.bz/orYv\">Build a Large Language Model From Scratch</a> 书籍的补充代码<br>",
    "<br>Code repository: <a href=\"https://github.com/rasbt/LLMs-from-scratch\">https://github.com/rasbt/LLMs-from-scratch</a>": "<br>代码仓库: <a href=\"https://github.com/rasbt/LLMs-from-scratch\">https://github.com/rasbt/LLMs-from-scratch</a>",
    "# Gemma 3 270M From Scratch (A Standalone Notebook)": "# 从头开始实现 Gemma 3 270M (独立笔记本)",
    "- This notebook is purposefully minimal and focuses on the code to re-implement Gemma 3 270M in pure PyTorch without relying on other external LLM libraries": "- 本笔记本特意保持极简，专注于在纯 PyTorch 中重新实现 Gemma 3 270M 的代码，而不依赖其他外部 LLM 库",
    "- For more information, see the official [Gemma 3 270M model card](https://huggingface.co/google/gemma-3-270m)": "- 有关更多信息，请参阅官方 [Gemma 3 270M 模型卡](https://huggingface.co/google/gemma-3-270m)",
    "- Below is a side-by-side comparison with Qwen3 0.6B as a reference model; if you are interested in the Qwen3 0.6B standalone notebook, you can find it [here](../11_qwen3)": "- 下面是与 Qwen3 0.6B 作为参考模型的并排比较；如果您对 Qwen3 0.6B 独立笔记本感兴趣，可以在 [这里](../11_qwen3) 找到它",
    "- About the code:": "- 关于代码：",
    "  - all code is my own code, mapping the Gemma 3 architecture onto the model code implemented in my [Build A Large Language Model (From Scratch)](http://mng.bz/orYv) book; the code is released under a permissive open-source Apache 2.0 license (see [LICENSE.txt](https://github.com/rasbt/LLMs-from-scratch/blob/main/LICENSE.txt))": "  - 所有代码均为我编写，将 Gemma 3 架构映射到我的 [Build A Large Language Model (From Scratch)](http://mng.bz/orYv) 书中实现与模型代码上；代码在宽松的开源 Apache 2.0 许可下发布（请参阅 [LICENSE.txt](https://github.com/rasbt/LLMs-from-scratch/blob/main/LICENSE.txt)）",
    "- This notebook supports both the base model and the instructmodel; which model to use can be controlled via the following flag:": "- 本笔记本支持基础模型和指令模型；可以通过以下标志控制使用哪个模型：",
    "# 1. Architecture code": "# 1. 架构代码",
    "# 2. Initialize model": "# 2. 初始化模型",
    "- A quick check that the forward pass works before continuing:": "- 在继续之前快速检查前向传递是否工作：",
    "# 3. Load pretrained weights": "# 3. 加载预训练权重",
    "- Please note that Google requires that you accept the Gemma 3 licensing terms before you can download the files; to do this, you have to create a Hugging Face Hub account and visit the [google/gemma-3-270m](https://huggingface.co/google/gemma-3-270m) repository to accept the terms": "- 请注意，Google 要求您在下载文件之前接受 Gemma 3 许可条款；为此，您必须创建一个 Hugging Face Hub 帐户并访问 [google/gemma-3-270m](https://huggingface.co/google/gemma-3-270m) 存储库以接受条款",
    "- Next, you will need to create an access token; to generate an access token with READ permissions, click on the profile picture in the upper right and click on \"Settings\"": "- 接下来，您需要创建一个访问令牌；要生成具有读取权限的访问令牌，请单击右上角的个人资料图片，然后单击“Settings”",
    "- Then, create and copy the access token so you can copy & paste it into the next code cell": "- 然后，创建并复制访问令牌，以便您可以将其复制并粘贴到下一个代码单元中",
    "# Uncomment and run the following code if you are executing the notebook for the first time": "# 如果您是第一次执行笔记本，请取消注释并运行以下代码",
    "# 4. Load tokenizer": "# 4. 加载分词器",
    "# 5. Generate text": "# 5. 生成文本",
    "# Optionally use torch.compile for an extra speed-up": "# 可选地使用 torch.compile 以获得额外的加速",
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
