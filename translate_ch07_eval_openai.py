# -*- coding: utf-8 -*-
import json
import os

def translate_notebook(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    translations = {
        "Supplementary code for the <a href=\"http://mng.bz/orYv\">Build a Large Language Model From Scratch</a> book by <a href=\"https://sebastianraschka.com\">Sebastian Raschka</a><br>\n": "《<a href=\"http://mng.bz/orYv\">从头开始构建大型语言模型</a>》一书的补充代码，作者 <a href=\"https://sebastianraschka.com\">Sebastian Raschka</a><br>\n",
        "<br>Code repository: <a href=\"https://github.com/rasbt/LLMs-from-scratch\">https://github.com/rasbt/LLMs-from-scratch</a>\n": "<br>代码仓库：<a href=\"https://github.com/rasbt/LLMs-from-scratch\">https://github.com/rasbt/LLMs-from-scratch</a>\n",
        "# Evaluating Instruction Responses Using the OpenAI API": "# 使用 OpenAI API 评估指令响应",
        "- This notebook uses OpenAI's GPT-4 API to evaluate responses by a instruction finetuned LLMs based on an dataset in JSON format that includes the generated model responses, for example:\n": "- 本笔记本使用 OpenAI 的 GPT-4 API 根据 JSON 格式的数据集评估指令微调 LLM 的响应，该数据集包含生成的模型响应，例如：\n",
        "## Test OpenAI API": "## 测试 OpenAI API",
        "- First, let's test if the OpenAI API is correctly set up\n": "- 首先，让我们测试 OpenAI API 是否已正确设置\n",
        "- If you don't have an account yet, you need to create one at https://platform.openai.com/\n": "- 如果您还没有帐户，则需要在 https://platform.openai.com/ 创建一个\n",
        "- Note that you will also have to transfer some funds to your account as the GPT-4 API is not free (see https://platform.openai.com/settings/organization/billing/overview)\n": "- 请注意，您还必须向您的帐户转账，因为 GPT-4 API 不是免费的（请参阅 https://platform.openai.com/settings/organization/billing/overview）\n",
        "- Running the experiments and creating the ~200 evaluations using the code in this notebook costs about $0.26 (26 cents) as of this writing": "- 截至撰写本文时，运行实验并使用本笔记本中的代码创建约 200 个评估大约需要 0.26 美元（26 美分）",
        "- First, we need to provide our OpenAI API secret key, which can be found at https://platform.openai.com/api-keys\n": "- 首先，我们需要提供我们的 OpenAI API 密钥，可以在 https://platform.openai.com/api-keys 找到\n",
        "- Make sure not to share this key with anyone\n": "- 确保不要与任何人共享此密钥\n",
        "- Add this secret key (`\"sk-...\"`) to the `config.json` file in this folder": "- 将此密钥 (`\"sk-...\"`) 添加到此文件夹中的 `config.json` 文件中",
        "- First, let's try the API with a simple example to make sure it works as intended:": "- 首先，让我们用一个简单的例子来尝试 API，以确保它按预期工作：",
        "## Load JSON Entries": "## 加载 JSON 条目",
        "- Here, we assume that we saved the test dataset and the model responses as a JSON file that we can load as follows:": "- 在这里，我们假设我们将测试数据集和模型响应保存为 JSON 文件，我们可以按如下方式加载它：",
        "- The structure of this file is as follows, where we have the given response in the test dataset (`'output'`) and responses by two different models (`'model 1 response'` and `'model 2 response'`):": "- 该文件的结构如下，其中我们在测试数据集中有给定的响应 (`'output'`) 以及两个不同模型的响应 (`'model 1 response'` 和 `'model 2 response'`)：",
        "- Below is a small utility function that formats the input for visualization purposes later:": "- 下面是一个小的实用函数，用于格式化输入以便稍后进行可视化：",
        "- Now, let's try the OpenAI API to compare the model responses (we only evaluate the first 5 responses for a visual comparison):": "- 现在，让我们尝试使用 OpenAI API 来比较模型响应（我们仅评估前 5 个响应以进行视觉比较）：",
        "- Note that the responses are very verbose; to quantify which model is better, we only want to return the scores:": "- 请注意，响应非常冗长；为了量化哪个模型更好，我们只想返回分数：",
        "- Please note that the response scores may vary because OpenAI's GPT models are not deterministic despite setting a random number seed, etc.": "- 请注意，尽管设置了随机数种子等，但响应分数可能会有所不同，因为 OpenAI 的 GPT 模型不是确定性的。",
        "- Let's now apply this evaluation to the whole dataset and compute the average score of each model:": "- 让我们现在将此评估应用于整个数据集并计算每个模型的平均分数：",
        "- Based on the evaluation above, we can say that the 1st model is substantially better than the 2nd model": "- 根据上面的评估，我们可以说第一个模型明显优于第二个模型"
    }

    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            for i, line in enumerate(cell['source']):
                if line in translations:
                    cell['source'][i] = translations[line]

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

if __name__ == "__main__":
    translate_notebook(
        '/Users/richard/Git/LLMs-from-scratch/ch07/03_model-evaluation/llm-instruction-eval-openai.ipynb',
        '/Users/richard/Git/LLMs-from-scratch/ch07/03_model-evaluation/llm-instruction-eval-openai_zh.ipynb'
    )
