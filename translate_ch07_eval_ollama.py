# -*- coding: utf-8 -*-
import json
import os

def translate_notebook(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    translations = {
        "Supplementary code for the <a href=\"http://mng.bz/orYv\">Build a Large Language Model From Scratch</a> book by <a href=\"https://sebastianraschka.com\">Sebastian Raschka</a><br>\n": "《<a href=\"http://mng.bz/orYv\">从头开始构建大型语言模型</a>》一书的补充代码，作者 <a href=\"https://sebastianraschka.com\">Sebastian Raschka</a><br>\n",
        "<br>Code repository: <a href=\"https://github.com/rasbt/LLMs-from-scratch\">https://github.com/rasbt/LLMs-from-scratch</a>\n": "<br>代码仓库：<a href=\"https://github.com/rasbt/LLMs-from-scratch\">https://github.com/rasbt/LLMs-from-scratch</a>\n",
        "# Evaluating Instruction Responses Locally Using a Llama 3 Model Via Ollama": "# 使用 Llama 3 模型通过 Ollama 在本地评估指令响应",
        "- This notebook uses an 8-billion-parameter Llama 3 model through ollama to evaluate responses of instruction finetuned LLMs based on a dataset in JSON format that includes the generated model responses, for example:\n": "- 本笔记本通过 ollama 使用 80 亿参数的 Llama 3 模型，根据包含生成的模型响应的 JSON 格式数据集评估指令微调 LLM 的响应，例如：\n",
        "- The code doesn't require a GPU and runs on a laptop (it was tested on a M3 MacBook Air)": "- 该代码不需要 GPU，可以在笔记本电脑上运行（已在 M3 MacBook Air 上测试）",
        "## Installing Ollama and Downloading Llama 3": "## 安装 Ollama 并下载 Llama 3",
        "- Ollama is an application to run LLMs efficiently\n": "- Ollama 是一个高效运行 LLM 的应用程序\n",
        "- It is a wrapper around [llama.cpp](https://github.com/ggerganov/llama.cpp), which implements LLMs in pure C/C++ to maximize efficiency\n": "- 它是 [llama.cpp](https://github.com/ggerganov/llama.cpp) 的包装器，后者用纯 C/C++ 实现 LLM 以最大限度地提高效率\n",
        "- Note that it is a tool for using LLMs to generate text (inference), not training or finetuning LLMs\n": "- 请注意，它是一个使用 LLM 生成文本（推理）的工具，而不是训练 or 微调 LLM\n",
        "- Prior to running the code below, install ollama by visiting [https://ollama.com](https://ollama.com) and following the instructions (for instance, clicking on the \"Download\" button and downloading the ollama application for your operating system)": "- 在运行下面的代码之前，请访问 [https://ollama.com](https://ollama.com) 并按照说明安装 ollama（例如，单击“下载”按钮并下载适用于您的操作系统的 ollama 应用程序）",
        "- For macOS and Windows users, click on the ollama application you downloaded; if it prompts you to install the command line usage, say \"yes\"\n": "- 对于 macOS 和 Windows 用户，单击您下载的 ollama 应用程序；如果它提示您安装命令行用法，请说“是”\n",
        "- Linux users can use the installation command provided on the ollama website\n": "- Linux 用户可以使用 ollama 网站上提供的安装命令\n",
        "- In general, before we can use ollama from the command line, we have to either start the ollama application or run `ollama serve` in a separate terminal\n": "- 通常，在我们从命令行使用 ollama 之前，我们要么启动 ollama 应用程序，要么在单独的终端中运行 `ollama serve`\n",
        "- With the ollama application or `ollama serve` running, in a different terminal, on the command line, execute the following command to try out the 8-billion-parameter Llama 3 model (the model, which takes up 4.7 GB of storage space, will be automatically downloaded the first time you execute this command)\n": "- 在运行 ollama 应用程序或 `ollama serve` 的情况下，在另一个终端的命令行中，执行以下命令以试用 80 亿参数的 Llama 3 模型（该模型占用 4.7 GB 的存储空间，将在您首次执行此命令时自动下载）\n",
        "The output looks like as follows:\n": "输出如下所示：\n",
        "- Note that `llama3` refers to the instruction finetuned 8-billion-parameter Llama 3 model\n": "- 请注意，`llama3` 指的是指令微调的 80 亿参数 Llama 3 模型\n",
        "- Alternatively, you can also use the larger 70-billion-parameter Llama 3 model, if your machine supports it, by replacing `llama3` with `llama3:70b`\n": "- 或者，如果您的机器支持，您也可以通过将 `llama3` 替换为 `llama3:70b` 来使用更大的 700 亿参数 Llama 3 模型\n",
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
        "- Now, let's get to the data evaluation part\n": "- 现在，让我们进入数据评估部分\n",
        "- Here, we assume that we saved the test dataset and the model responses as a JSON file that we can load as follows:": "- 在这里，我们假设我们将测试数据集和模型响应保存为 JSON 文件，我们可以按如下方式加载它：",
        "- The structure of this file is as follows, where we have the given response in the test dataset (`'output'`) and responses by two different models (`'model 1 response'` and `'model 2 response'`):": "- 该文件的结构如下，其中我们在测试数据集中有给定的响应 (`'output'`) 以及两个不同模型的响应 (`'model 1 response'` 和 `'model 2 response'`)：",
        "- Below is a small utility function that formats the input for visualization purposes later:": "- 下面是一个小的实用函数，用于格式化输入以便稍后进行可视化：",
        "- Now, let's try the ollama API to compare the model responses (we only evaluate the first 5 responses for a visual comparison):": "- 现在，让我们尝试使用 ollama API 来比较模型响应（我们仅评估前 5 个响应以进行视觉比较）：",
        "- Note that the responses are very verbose; to quantify which model is better, we only want to return the scores:": "- 请注意，响应非常冗长；为了量化哪个模型更好，我们只想返回分数：",
        "- Let's now apply this evaluation to the whole dataset and compute the average score of each model (this takes about 1 minute per model on an M3 MacBook Air laptop)\n": "- 让我们现在将此评估应用于整个数据集并计算每个模型的平均分数（在 M3 MacBook Air 笔记本电脑上每个模型大约需要 1 分钟）\n",
        "- Note that ollama is not fully deterministic across operating systems (as of this writing) so the numbers you are getting might slightly differ from the ones shown below": "- 请注意，ollama 在不同的操作系统上并不是完全确定性的（在撰写本文时），因此您获得的数字可能与下面显示的数字略有不同",
        "- Based on the evaluation above, we can say that the 1st model is better than the 2nd model": "- 根据上面的评估，我们可以说第一个模型优于第二个模型"
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
        '/Users/richard/Git/LLMs-from-scratch/ch07/03_model-evaluation/llm-instruction-eval-ollama.ipynb',
        '/Users/richard/Git/LLMs-from-scratch/ch07/03_model-evaluation/llm-instruction-eval-ollama_zh.ipynb'
    )
