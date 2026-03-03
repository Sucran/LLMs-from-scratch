# -*- coding: utf-8 -*-
import json

# Input and output file paths
input_file = "ch05/14_ch05_with_other_llms/ch05-llama32.ipynb"
output_file = "ch05/14_ch05_with_other_llms/ch05-llama32_zh.ipynb"

# Translation mapping
translations = {
    "Supplementary code for the <a href=\"http://mng.bz/orYv\">Build a Large Language Model From Scratch</a> book by <a href=\"https://sebastianraschka.com\">Sebastian Raschka</a><br>": "Sebastian Raschka 的 <a href=\"http://mng.bz/orYv\">Build a Large Language Model From Scratch</a> 书籍的补充代码<br>",
    "<br>Code repository: <a href=\"https://github.com/rasbt/LLMs-from-scratch\">https://github.com/rasbt/LLMs-from-scratch</a>": "<br>代码仓库: <a href=\"https://github.com/rasbt/LLMs-from-scratch\">https://github.com/rasbt/LLMs-from-scratch</a>",
    "# Chapter 5 Bonus: Pretraining Llama 3 on Unlabeled Data": "# 第 5 章额外内容：在未标记数据上预训练 Llama 3",
    "- This notebook plugs in the [Llama 3 1B from-scratch](../07_gpt_to_llama/standalone-llama32.ipynb) model into (the pretraining portion) of chapter 5": "- 本笔记本将 [Llama 3 1B from-scratch](../07_gpt_to_llama/standalone-llama32_zh.ipynb) 模型插入到第 5 章的（预训练部分）",
    "- This is to show how to use Llama 3 1B as a drop-in replacement for the GTP-2 model used in [chapter 5](../01_main-chapter-code/ch05.ipynb)": "- 这是为了展示如何使用 Llama 3 1B 作为 [第 5 章](../01_main-chapter-code/ch05_zh.ipynb) 中使用的 GPT-2 模型的直接替代品",
    "## 5.1 Evaluating generative text models": "## 5.1 评估生成文本模型",
    "- No code": "- 无代码",
    "### 5.1.1 Using Llama 3 to generate text": "### 5.1.1 使用 Llama 3 生成文本",
    "### 5.1.2 Calculating the text generation loss: cross-entropy and perplexity": "### 5.1.2 计算文本生成损失：交叉熵和困惑度",
    "- Similar to chapter 5": "- 与第 5 章类似",
    "### 5.1.3 Calculating the training and validation set losses": "### 5.1.3 计算训练和验证集损失",
    "- A quick check that the text loaded ok by printing the first and last 99 characters": "- 通过打印前 99 个和最后 99 个字符来快速检查文本是否加载正常",
    "- An optional check that the data was loaded correctly:": "- 检查数据是否正确加载的可选步骤：",
    "- Another optional check that the token sizes are in the expected ballpark:": "- 另一个可选检查，确保标记大小在预期范围内：",
    "- Next, we implement a utility function to calculate the cross-entropy loss of a given batch": "- 接下来，我们要实现一个实用函数来计算给定批次的交叉熵损失",
    "- In addition, we implement a second utility function to compute the loss for a user-specified number of batches in a data loader": "- 此外，我们实现第二个实用函数来计算数据加载器中用户指定批次数量的损失",
    "- If you have a machine with a CUDA-supported GPU, the LLM will train on the GPU without making any changes to the code": "- 如果您有一台支持 CUDA GPU 的机器，LLM 将在 GPU 上训练，无需更改任何代码",
    "- Via the `device` setting, we ensure that the data is loaded onto the same device as the LLM model": "- 通过 `device` 设置，我们确保数据加载到与 LLM 模型相同的设备上",
    "## 5.2 Training an LLM": "## 5.2 训练 LLM",
    "## 5.3 Decoding strategies to control randomness": "## 5.3 控制随机性的解码策略",
    "### 5.3.1 Temperature scaling": "### 5.3.1 温度缩放",
    "### 5.3.2 Top-k sampling": "### 5.3.2 Top-k 采样",
    "### 5.3.3 Modifying the text generation function": "### 5.3.3 修改文本生成函数",
    "## 5.4 Loading and saving model weights in PyTorch": "## 5.4 在 PyTorch 中加载和保存模型权重",
    "## 5.5 Loading pretrained weights": "## 5.5 加载预训练权重",
    "- See [Qwen3 0.6B from-scratch](../11_qwen3/standalone-qwen3.ipynb)": "- 参见 [Qwen3 0.6B from-scratch](../11_qwen3/standalone-qwen3_zh.ipynb)",
    "## Summary and takeaways": "## 总结和要点",
    "- Skipped": "- 跳过",
    "ch05.ipynb": "ch05_zh.ipynb",
    "standalone-llama32.ipynb": "standalone-llama32_zh.ipynb",
    "standalone-qwen3.ipynb": "standalone-qwen3_zh.ipynb"
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
