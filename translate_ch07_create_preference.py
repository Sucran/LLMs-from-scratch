# -*- coding: utf-8 -*-
import json
import os

def translate_notebook(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    translations = {
        "Supplementary code for the <a href=\"http://mng.bz/orYv\">Build a Large Language Model From Scratch</a> book by <a href=\"https://sebastianraschka.com\">Sebastian Raschka</a><br>\n": "《<a href=\"http://mng.bz/orYv\">从头开始构建大型语言模型</a>》一书的补充代码，作者 <a href=\"https://sebastianraschka.com\">Sebastian Raschka</a><br>\n",
        "<br>Code repository: <a href=\"https://github.com/rasbt/LLMs-from-scratch\">https://github.com/rasbt/LLMs-from-scratch</a>\n": "<br>代码仓库：<a href=\"https://github.com/rasbt/LLMs-from-scratch\">https://github.com/rasbt/LLMs-from-scratch</a>\n",
        "# Generating A Preference Dataset With Llama 3.1 70B And Ollama": "# 使用 Llama 3.1 70B 和 Ollama 生成偏好数据集",
        "- Preference finetuning is a process to align an instruction-finetuned LLM with human preferences\n": "- 偏好微调是将指令微调的 LLM 与人类偏好对齐的过程\n",
        "- There are multiple ways to create a dataset for preference finetuning an LLM\n": "- 有多种方法可以为偏好微调 LLM 创建数据集\n",
        "  1. We use the instruction-finetuned LLM to generate multiple responses and have humans rank them based on their preference and/or given preference criteria\n": "  1. 我们使用指令微调的 LLM 生成多个响应，并让人们根据他们的偏好和/或给定的偏好标准对它们进行排名\n",
        "  2. We use the instruction-finetuned LLM to generate multiple responses and have LLMs rank them based on given preference criteria\n": "  2. 我们使用指令微调的 LLM 生成多个响应，并让 LLM 根据给定的偏好标准对它们进行排名\n",
        "  3. We use an LLM to generate preferred and dispreferred responses given certain preference criteria\n": "  3. 我们使用 LLM 根据某些偏好标准生成偏好和不偏好的响应\n",
        "- In this notebook, we consider approach 3\n": "- 在本笔记本中，我们考虑方法 3\n",
        "- This notebook uses a 70-billion-parameter Llama 3.1-Instruct model through ollama to generate preference labels for an instruction dataset\n": "- 本笔记本通过 ollama 使用 700 亿参数的 Llama 3.1-Instruct 模型为指令数据集生成偏好标签\n",
        "- The expected format of the instruction dataset is as follows:\n": "- 指令数据集的预期格式如下：\n",
        "### Input\n": "### 输入\n",
        "The output dataset will look as follows, where more polite responses are preferred (`'chosen'`), and more impolite responses are dispreferred (`'rejected'`):\n": "输出数据集如下所示，其中更礼貌的响应是首选 (`'chosen'`)，更不礼貌的响应是不受欢迎的 (`'rejected'`)：\n",
        "### Output\n": "### 输出\n",
        "- The code doesn't require a GPU and runs on a laptop given enough RAM": "- 代码不需要 GPU，只要有足够的 RAM，就可以在笔记本电脑上运行",
        "## Installing Ollama and Downloading Llama 3.1": "## 安装 Ollama 并下载 Llama 3.1",
        "- Ollama is an application to run LLMs efficiently\n": "- Ollama 是一个高效运行 LLM 的应用程序\n",
        "- It is a wrapper around [llama.cpp](https://github.com/ggerganov/llama.cpp), which implements LLMs in pure C/C++ to maximize efficiency\n": "- 它是 [llama.cpp](https://github.com/ggerganov/llama.cpp) 的包装器，后者用纯 C/C++ 实现 LLM 以最大限度地提高效率\n",
        "- Note that it is a tool for using LLMs to generate text (inference), not training or finetuning LLMs\n": "- 请注意，它是一个使用 LLM 生成文本（推理）的工具，而不是训练或微调 LLM\n",
        "- Prior to running the code below, install ollama by visiting [https://ollama.com](https://ollama.com) and following the instructions (for instance, clicking on the \"Download\" button and downloading the ollama application for your operating system)": "- 在运行下面的代码之前，请访问 [https://ollama.com](https://ollama.com) 并按照说明安装 ollama（例如，单击“下载”按钮并下载适用于您的操作系统的 ollama 应用程序）",
        "- For macOS and Windows users, click on the ollama application you downloaded; if it prompts you to install the command line usage, say \"yes\"\n": "- 对于 macOS 和 Windows 用户，单击您下载的 ollama 应用程序；如果它提示您安装命令行用法，请说“是”\n",
        "- Linux users can use the installation command provided on the ollama website\n": "- Linux 用户可以使用 ollama 网站上提供的安装命令\n",
        "- In general, before we can use ollama from the command line, we have to either start the ollama application or run `ollama serve` in a separate terminal\n": "- 通常，在我们从命令行使用 ollama 之前，我们要么启动 ollama 应用程序，要么在单独的终端中运行 `ollama serve`\n",
        "- With the ollama application or `ollama serve` running, in a different terminal, on the command line, execute the following command to try out the 70-billion-parameter Llama 3.1 model \n": "- 在运行 ollama 应用程序或 `ollama serve` 的情况下，在另一个终端的命令行中，执行以下命令以试用 700 亿参数的 Llama 3.1 模型\n",
        "The output looks like as follows:\n": "输出如下所示：\n",
        "- Note that `llama3.1:70b` refers to the instruction finetuned 70-billion-parameter Llama 3.1 model\n": "- 请注意，`llama3.1:70b` 指的是指令微调的 700 亿参数 Llama 3.1 模型\n",
        "- Alternatively, you can also use the smaller, more resource-effiicent 8-billion-parameters Llama 3.1 model, by replacing `llama3.1:70b` with `llama3.1`\n": "- 或者，您也可以通过将 `llama3.1:70b` 替换为 `llama3.1` 来使用更小、更节省资源的 80 亿参数 Llama 3.1 模型\n",
        "- After the download has been completed, you will see a command line prompt that allows you to chat with the model\n": "- 下载完成后，您将看到一个命令行提示符，允许您与模型聊天\n",
        "- Try a prompt like \"What do llamas eat?\", which should return an output similar to the following:\n": "- 尝试诸如“What do llamas eat?”之类的提示，它应该返回类似于以下的输出：\n",
        "- You can end this session using the input `/bye`": "- 您可以使用输入 `/bye` 结束此会话",
        "## Using Ollama's REST API": "## 使用 Ollama 的 REST API",
        "- Now, an alternative way to interact with the model is via its REST API in Python via the following function\n": "- 现在，与模型交互的另一种方法是通过 Python 中的 REST API，通过以下函数\n",
        "- Before you run the next cells in this notebook, make sure that ollama is still running, as described above, via\n": "- 在运行本笔记本中的下一个单元格之前，请确保 ollama 仍在运行，如上所述，通过\n",
        "  - `ollama serve` in a terminal\n": "  - 终端中的 `ollama serve`\n",
        "  - the ollama application\n": "  - ollama 应用程序\n",
        "- Next, run the following code cell to query the model": "- 接下来，运行以下代码单元格以查询模型",
        "- First, let's try the API with a simple example to make sure it works as intended:": "- 首先，让我们用一个简单的例子来尝试 API，以确保它按预期工作：",
        "## Load JSON Entries": "## 加载 JSON 条目",
        "- Now, let's get to the data generation part\n": "- 现在，让我们进入数据生成部分\n",
        "- Here, for a hands-on example, we use the `instruction-data.json` file that we originally used to instruction-finetune the model in chapter 7:": "- 在这里，作为一个动手示例，我们使用我们最初在第 7 章中用于指令微调模型的 `instruction-data.json` 文件：",
        "- The structure of this file is as follows, where we have the given response in the test dataset (`'output'`) that we trained the model to generate via instruction finetuning based on the `'input'` and `'instruction'`": "- 该文件的结构如下，其中我们在测试数据集中有给定的响应 (`'output'`)，我们训练模型通过基于 `'input'` 和 `'instruction'` 的指令微调来生成该响应",
        "- Below is a small utility function that formats the instruction and input:": "- 下面是一个小的实用函数，用于格式化指令和输入：",
        "- Now, let's try the ollama API to generate a `'chosen'` and `'rejected'` response for preference tuning a model\n": "- 现在，让我们尝试使用 ollama API 生成 `'chosen'` 和 `'rejected'` 响应以进行偏好调整模型\n",
        "- Here, to for illustration purposes, we create answers that are more or less polite\n": "- 在这里，为了说明目的，我们创建或多或少礼貌的答案\n",
        "- If we find that the generated responses above look reasonable, we can go to the next step and apply the prompt to the whole dataset\n": "- 如果我们发现上面生成的响应看起来很合理，我们可以进入下一步并将提示应用于整个数据集\n",
        "- Here, we add a `'chosen'` key for the preferred response and a `'rejected'` response for the dispreferred response": "- 在这里，我们为首选响应添加一个 `'chosen'` 键，为不首选响应添加一个 `'rejected'` 响应",
        "- Let's now apply this evaluation to the whole dataset and compute the average score of each model (this takes about 1 minute per model on an M3 MacBook Air laptop)\n": "- 让我们现在将此评估应用于整个数据集并计算每个模型的平均分数（在 M3 MacBook Air 笔记本电脑上每个模型大约需要 1 分钟）\n",
        "- Note that ollama is not fully deterministic across operating systems (as of this writing) so the numbers you are getting might slightly differ from the ones shown below": "- 请注意，ollama 在不同的操作系统上并不是完全确定性的（在撰写本文时），因此您获得的数字可能与下面显示的数字略有不同"
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
        '/Users/richard/Git/LLMs-from-scratch/ch07/04_preference-tuning-with-dpo/create-preference-data-ollama.ipynb',
        '/Users/richard/Git/LLMs-from-scratch/ch07/04_preference-tuning-with-dpo/create-preference-data-ollama_zh.ipynb'
    )
