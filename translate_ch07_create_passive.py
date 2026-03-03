# -*- coding: utf-8 -*-
import json
import os

def translate_notebook(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    translations = {
        "Supplementary code for the <a href=\"http://mng.bz/orYv\">Build a Large Language Model From Scratch</a> book by <a href=\"https://sebastianraschka.com\">Sebastian Raschka</a><br>\n": "《<a href=\"http://mng.bz/orYv\">从头开始构建大型语言模型</a>》一书的补充代码，作者 <a href=\"https://sebastianraschka.com\">Sebastian Raschka</a><br>\n",
        "<br>Code repository: <a href=\"https://github.com/rasbt/LLMs-from-scratch\">https://github.com/rasbt/LLMs-from-scratch</a>\n": "<br>代码仓库：<a href=\"https://github.com/rasbt/LLMs-from-scratch\">https://github.com/rasbt/LLMs-from-scratch</a>\n",
        "# Create \"Passive Voice\" Entries for an Instruction Dataset": "# 为指令数据集创建“被动语态”条目",
        "- This notebook uses OpenAI's GPT-4 to create \"passive voice\" entries for an instruction dataset, as shown in the example below\n": "- 本笔记本使用 OpenAI 的 GPT-4 为指令数据集创建“被动语态”条目，如下例所示\n",
        "## 1. Test OpenAI API": "## 1. 测试 OpenAI API",
        "- First, let's test if the OpenAI API is correctly set up\n": "- 首先，让我们测试 OpenAI API 是否已正确设置\n",
        "- If you don't have an account yet, you need to create one at https://platform.openai.com/\n": "- 如果您还没有帐户，则需要在 https://platform.openai.com/ 创建一个\n",
        "- Note that you will also have to transfer some funds to your account as the GPT-4 API is not free (see https://platform.openai.com/settings/organization/billing/overview)\n": "- 请注意，您还必须向您的帐户转账，因为 GPT-4 API 不是免费的（请参阅 https://platform.openai.com/settings/organization/billing/overview）\n",
        "- Creating the ~200 passive voice entries using the code in this notebook costs about $0.13 (13 cents)": "- 使用本笔记本中的代码创建约 200 个被动语态条目大约需要 0.13 美元（13 美分）",
        "- First, we need to provide our OpenAI API secret key, which can be found at https://platform.openai.com/api-keys\n": "- 首先，我们需要提供我们的 OpenAI API 密钥，可以在 https://platform.openai.com/api-keys 找到\n",
        "- Make sure not to share this key with anyone\n": "- 确保不要与任何人共享此密钥\n",
        "- Add this secret key (`\"sk-...\"`) to the `config.json` file in this folder": "- 将此密钥 (`\"sk-...\"`) 添加到此文件夹中的 `config.json` 文件中",
        "- First, let's try the API with a simple example to make sure it works as intended:": "- 首先，让我们用一个简单的例子来尝试 API，以确保它按预期工作：",
        "## 2. Create JSON Entries": "## 2. 创建 JSON 条目",
        "- Next, we load the file we want to modify:": "- 接下来，我们加载要修改的文件：",
        "- And we try the OpenAI chat API on a small sample first to ensure that it works correctly:": "- 我们首先在一个小样本上尝试 OpenAI 聊天 API，以确保它正常工作：",
        "- Let's now extend the code to add the generated entries to the `json_data` and add a progress bar:": "- 现在让我们扩展代码，将生成的条目添加到 `json_data` 中并添加进度条：",
        "- One more time, let's make sure that the new entries (`\"output_2\"`) look ok": "- 再一次，让我们确保新条目 (`\"output_2\"`) 看起来没问题",
        "- Finally, if everything above looks ok, let's run the conversion to passive voice on our entire json dataset (this takes about 3 minutes):": "- 最后，如果上面的一切看起来都很好，让我们对整个 json 数据集运行转换为被动语态（这大约需要 3 分钟）：",
        "- After the conversion is completed, we save the file:": "- 转换完成后，我们保存文件："
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
        '/Users/richard/Git/LLMs-from-scratch/ch07/02_dataset-utilities/create-passive-voice-entries.ipynb',
        '/Users/richard/Git/LLMs-from-scratch/ch07/02_dataset-utilities/create-passive-voice-entries_zh.ipynb'
    )
