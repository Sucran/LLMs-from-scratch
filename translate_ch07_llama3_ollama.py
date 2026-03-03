
# -*- coding: utf-8 -*-
import json
import os

def translate_notebook(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    translations = {
        "# Generating An Instruction Dataset via Llama 3 and Ollama": "# 通过 Llama 3 和 Ollama 生成指令数据集",
        "- This notebook uses an 8-billion-parameter Llama 3 model through ollama to generate a synthetic dataset using the \"hack\" proposed in the \"Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing\" paper ([https://arxiv.org/abs/2406.08464](https://arxiv.org/abs/2406.08464))\n": "- 本笔记本通过 ollama 使用 80 亿参数的 Llama 3 模型，利用“Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing”论文 ([https://arxiv.org/abs/2406.08464](https://arxiv.org/abs/2406.08464)) 中提出的“hack”方法来生成合成数据集。\n",
        "- The generated dataset will be an instruction dataset with \"instruction\" and \"output\" field similar to what can be found in Alpaca:\n": "- 生成的数据集将是一个包含“instruction”和“output”字段的指令数据集，类似于 Alpaca 中的数据集：\n",
        "- The code doesn't require a GPU and runs on a laptop (it was tested on a M3 MacBook Air)\n": "- 该代码不需要 GPU，可以在笔记本电脑上运行（已在 M3 MacBook Air 上测试）。\n",
        "*Note that the instruction datasets created here are for educational purposes. However, it is the users' duty to ensure that their use adheres to the terms of the relevant licensing agreements with Meta AI's Llama 3.*": "*请注意，此处创建的指令数据集仅用于教育目的。但是，用户有责任确保其使用符合 Meta AI Llama 3 的相关许可协议条款。*",
        "## Installing Ollama and Downloading Llama 3": "## 安装 Ollama 并下载 Llama 3",
        "- Ollama is an application to run LLMs efficiently\n": "- Ollama 是一个可以高效运行 LLM 的应用程序。\n",
        "- It is a wrapper around [llama.cpp](https://github.com/ggerganov/llama.cpp), which implements LLMs in pure C/C++ to maximize efficiency\n": "- 它是 [llama.cpp](https://github.com/ggerganov/llama.cpp) 的封装器，后者用纯 C/C++ 实现 LLM 以最大化效率。\n",
        "- Note that it is a tool for using LLMs to generate text (inference), not training or finetuning LLMs\n": "- 请注意，它是一个使用 LLM 生成文本（推理）的工具，而不是用于训练或微调 LLM。\n",
        "- Prior to running the code below, install ollama by visiting [https://ollama.com](https://ollama.com) and following the instructions (for instance, clicking on the \"Download\" button and downloading the ollama application for your operating system)": "- 在运行下面的代码之前，请访问 [https://ollama.com](https://ollama.com) 并按照说明安装 ollama（例如，点击“Download”按钮并下载适用于您操作系统的 ollama 应用程序）。",
        "- For macOS and Windows users, click on the ollama application you downloaded; if it prompts you to install the command line usage, say \"yes\"\n": "- 对于 macOS 和 Windows 用户，点击您下载的 ollama 应用程序；如果它提示您安装命令行用法，请选择“是”。\n",
        "- Linux users can use the installation command provided on the ollama website\n": "- Linux 用户可以使用 ollama 网站上提供的安装命令。\n",
        "- In general, before we can use ollama from the command line, we have to either start the ollama application or run `ollama serve` in a separate terminal\n": "- 一般来说，在我们从命令行使用 ollama 之前，我们必须启动 ollama 应用程序或在单独的终端中运行 `ollama serve`。\n",
        "- With the ollama application or `ollama serve` running, in a different terminal, on the command line, execute the following command to try out the 8-billion-parameter Llama 3 model (the model, which takes up 4.7 GB of storage space, will be automatically downloaded the first time you execute this command)\n": "- 在 ollama 应用程序或 `ollama serve` 运行的情况下，在不同的终端中，在命令行上执行以下命令以试用 80 亿参数的 Llama 3 模型（该模型占用 4.7 GB 的存储空间，将在您首次执行此命令时自动下载）。\n",
        "- Note that `llama3` refers to the instruction finetuned 8-billion-parameter Llama 3 model\n": "- 请注意，`llama3` 指的是经过指令微调的 80 亿参数 Llama 3 模型。\n",
        "- Alternatively, you can also use the larger 70-billion-parameter Llama 3 model, if your machine supports it, by replacing `llama3` with `llama3:70b`\n": "- 或者，如果您的机器支持，您也可以通过将 `llama3` 替换为 `llama3:70b` 来使用更大的 700 亿参数 Llama 3 模型。\n",
        "- After the download has been completed, you will see a command line prompt that allows you to chat with the model\n": "- 下载完成后，您将看到一个允许您与模型聊天的命令行提示符。\n",
        "- Try a prompt like \"What do llamas eat?\", which should return an output similar to the following:\n": "- 尝试像“What do llamas eat?”这样的提示，它应该返回类似于以下的输出：\n",
        "- You can end this session using the input `/bye`": "- 您可以使用输入 `/bye` 结束此会话。",
        "## Using Ollama's REST API": "## 使用 Ollama 的 REST API",
        "- Now, an alternative way to interact with the model is via its REST API in Python via the following function\n": "- 现在，与模型交互的另一种方式是通过 Python 中的 REST API，通过以下函数。\n",
        "- Before you run the next cells in this notebook, make sure that ollama is still running, as described above, via\n": "- 在运行本笔记本中的下一个单元格之前，请确保 ollama 仍在运行，如上所述，通过\n",
        "  - `ollama serve` in a terminal\n": "  - 终端中的 `ollama serve`\n",
        "  - the ollama application\n": "  - ollama 应用程序\n",
        "- Next, run the following code cell to query the model": "- 接下来，运行以下代码单元格来查询模型",
        "- First, let's try the API with a simple example to make sure it works as intended:": "- 首先，让我们用一个简单的例子尝试 API，以确保它按预期工作：",
        "## Extract Instructions": "## 提取指令",
        "- Now, let's use the \"hack\" proposed in the paper: we provide the empty prompt template `\"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\"` prompt, which will cause the instruction-finetuned Llama 3 model to generate an instruction": "- 现在，让我们使用论文中提出的“hack”：我们提供空的提示模板 `\"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\"` 提示，这将导致经过指令微调的 Llama 3 模型生成指令。",
        "- As we can see above, surprisingly, the model indeed generated an instruction": "- 如上所示，令人惊讶的是，模型确实生成了一条指令。",
        "## Generate Responses": "## 生成回复",
        "- Now, the next step is to create the corresponding response, which can be done by simply passing the instruction as input": "- 现在，下一步是创建相应的回复，这可以通过简单地将指令作为输入传递来完成。",
        "## Generate Dataset": "## 生成数据集",
        "- We can scale up this approach to an arbitrary number of data samples (you may want to apply some optional filtering length or quality (e.g., using another LLM to rate the generated data)\n": "- 我们可以将这种方法扩展到任意数量的数据样本（您可能希望应用一些可选的长度或质量过滤（例如，使用另一个 LLM 对生成的数据进行评分））。\n",
        "- Below, we generate 5 synthetic instruction-response pairs, which takes about 3 minutes on an M3 MacBook Air\n": "- 下面，我们生成 5 个合成的指令-回复对，在 M3 MacBook Air 上大约需要 3 分钟。\n",
        "- (To generate a dataset suitable for instruction finetuning, we want to increase this to at least 1k to 50k and perhaps run it on a GPU to generate the examples in a more timely fashion)\n": "- （为了生成适合指令微调的数据集，我们希望将其增加到至少 1k 到 50k，并可能在 GPU 上运行它以更及时地生成示例）。\n",
        "**Tip**\n": "**提示**\n",
        "- You can generate even higher-quality responses by changing `model=\"llama3\"` to `model=\"llama3:70b\"`, however, this will require more computational resources": "- 您可以通过将 `model=\"llama3\"` 更改为 `model=\"llama3:70b\"` 来生成更高质量的回复，但这将需要更多的计算资源。"
    }

    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            new_source = []
            for line in cell['source']:
                translated_line = translations.get(line.strip(), line)
                if line.strip() in translations:
                     new_source.append(translated_line + ('\n' if line.endswith('\n') else ''))
                else:
                    new_source.append(line)
            cell['source'] = new_source

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

if __name__ == "__main__":
    translate_notebook('ch07/05_dataset-generation/llama3-ollama.ipynb', 'ch07/05_dataset-generation/llama3-ollama_zh.ipynb')
