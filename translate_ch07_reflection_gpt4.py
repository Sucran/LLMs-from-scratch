
# -*- coding: utf-8 -*-
import json
import os

def translate_notebook(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    translations = {
        "# Improving Instruction-Data Via Reflection-Tuning Using GPT-4": "# 使用 GPT-4 通过反思微调改进指令数据",
        "- This notebook uses OpenAI's GPT-4 API to implement the dataset refinement process from the [Reflection-Tuning: Data Recycling Improves LLM Instruction-Tuning](https://arxiv.org/abs/2310.11716) paper\n": "- 本笔记本使用 OpenAI 的 GPT-4 API 来实现 [Reflection-Tuning: Data Recycling Improves LLM Instruction-Tuning](https://arxiv.org/abs/2310.11716) 论文中的数据集细化过程。\n",
        "- In the original paper, the researchers refined the [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) and [WizardLM](https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_70k) instruction-finetuning datasets; in this notebook, we refine the [instruction-dataset used in chapter 7](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/instruction-data.json) (however, since it has the same format as Alpaca, the same code works with the Alpaca dataset as well)\n": "- 在原始论文中，研究人员改进了 [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) 和 [WizardLM](https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_70k) 指令微调数据集；在本笔记本中，我们改进了 [第 7 章中使用的指令数据集](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/instruction-data.json)（但是，由于它与 Alpaca 具有相同的格式，因此相同的代码也适用于 Alpaca 数据集）。\n",
        "- The expected dataset format is as follows:\n": "- 预期的数据集格式如下：\n",
        "> Please note that this notebook reproduces the approach from the paper in which the authors used the GPT API to enhance existing datasets. However, it's important to be aware that GPT API-generated data may not be used to develop models that compete with OpenAI, as specified in the [OpenAI Terms of Use](https://openai.com/policies/row-terms-of-use/): \"What you cannot do... Use Output to develop models that compete with OpenAI.\"\n": "> 请注意，本笔记本重现了论文中的方法，其中作者使用 GPT API 来增强现有数据集。但是，请务必注意，GPT API 生成的数据不得用于开发与 OpenAI 竞争的模型，正如 [OpenAI 使用条款](https://openai.com/policies/row-terms-of-use/) 中所指定的那样：“你不能做的事情……使用输出开发与 OpenAI 竞争的模型。”\n",
        "You can find a relevant discussion [here](https://www.reddit.com/r/LocalLLaMA/comments/17vbg1f/does_openai_tos_prohibit_generating_datasets_for/)).": "您可以在 [此处](https://www.reddit.com/r/LocalLLaMA/comments/17vbg1f/does_openai_tos_prohibit_generating_datasets_for/) 找到相关讨论。",
        "## Test OpenAI API": "## 测试 OpenAI API",
        "- First, let's test if the OpenAI API is correctly set up\n": "- 首先，让我们测试 OpenAI API 是否已正确设置。\n",
        "- If you don't have an account yet, you need to create one at https://platform.openai.com/\n": "- 如果您还没有帐户，则需要在 https://platform.openai.com/ 创建一个。\n",
        "- Note that you will also have to transfer some funds to your account as the GPT-4 API is not free (see https://platform.openai.com/settings/organization/billing/overview)\n": "- 请注意，您还必须向您的帐户转账，因为 GPT-4 API 不是免费的（请参阅 https://platform.openai.com/settings/organization/billing/overview）。\n",
        "- Running the code exactly as it appears in this notebook costs about \\$0.03 (3 cents) with GPT-4o-mini as of this writing\n": "- 截至撰写本文时，使用 GPT-4o-mini 运行本笔记本中的代码大约需要花费 0.03 美元（3 美分）。\n",
        "- Applying the two methodologies above to all 1100 entries in the chapter 7 instruction dataset costs about \\$0.60 (60 cents)": "- 将上述两种方法应用于第 7 章指令数据集中的所有 1100 个条目大约需要花费 0.60 美元（60 美分）。",
        "- First, we need to provide our OpenAI API secret key, which can be found at https://platform.openai.com/api-keys\n": "- 首先，我们需要提供我们的 OpenAI API 密钥，可以在 https://platform.openai.com/api-keys 找到。\n",
        "- Make sure not to share this key with anyone\n": "- 确保不要与任何人分享此密钥。\n",
        "- Add this secret key (`\"sk-...\"`) to the `config.json` file in this folder": "- 将此密钥 (`\"sk-...\"`) 添加到此文件夹中的 `config.json` 文件中。",
        "- First, let's try the API with a simple example to make sure it works as intended:": "- 首先，让我们用一个简单的例子尝试 API，以确保它按预期工作：",
        "## Load JSON Entries": "## 加载 JSON 条目",
        "- Next, let's load and process the instruction dataset\n": "- 接下来，让我们加载和处理指令数据集。\n",
        "- Here, we assume that we saved the test dataset and the model responses as a JSON file that we can load as follows:": "- 在这里，我们假设我们将测试数据集和模型回复保存为 JSON 文件，我们可以按如下方式加载：",
        "- Let's print one of the dataset entries to see its structure:": "- 让我们打印其中一个数据集条目以查看其结构：",
        "## Improve Instructions": "## 改进指令",
        "- The Reflection-Tuning authors shared two approaches: (1) improving the instructions and (2) improving the responses\n": "- Reflection-Tuning 作者分享了两种方法：（1）改进指令和（2）改进回复。\n",
        "- Let's begin by improving the instructions in a given dataset\n": "- 让我们从改进给定数据集中的指令开始。\n",
        "- Below is a small utility function from the [Reflection-Tuning repository](https://github.com/tianyi-lab/Reflection_Tuning/blob/main/reflection_code/reflect_response.py) to format the inputs to the GPT-4 model for this dataset refinement": "- 下面是一个来自 [Reflection-Tuning 仓库](https://github.com/tianyi-lab/Reflection_Tuning/blob/main/reflection_code/reflect_response.py) 的小实用函数，用于为此数据集细化格式化 GPT-4 模型的输入。",
        "- To see how it works, consider the dataset entry, `json_data[2]`": "- 要查看它是如何工作的，请考虑数据集条目 `json_data[2]`",
        "- We can refine the instruction as follows, using `build_instruction_reflection_prompt_no_input` function defined above:": "- 我们可以使用上面定义的 `build_instruction_reflection_prompt_no_input` 函数如下改进指令：",
        "- The response is very verbose, which is useful for analysis purposes; also, it helps the GPT-4 model to make improvements via the chain-of-thought prompting approach\n": "- 回复非常详细，这对于分析目的很有用；此外，它还有助于 GPT-4 模型通过思维链提示方法进行改进。\n",
        "- However, to construct the improved dataset, we are actually only interested in new instructions and outputs, not the analyses\n": "- 然而，为了构建改进的数据集，我们实际上只对新的指令和输出感兴趣，而不是分析。\n",
        "- We can use the following utility code from the [Reflection-Tuning repository](https://github.com/tianyi-lab/Reflection_Tuning/blob/main/reflection_code/reflect_response.py) to extract the model's improved instructions and outputs": "- 我们可以使用来自 [Reflection-Tuning 仓库](https://github.com/tianyi-lab/Reflection_Tuning/blob/main/reflection_code/reflect_response.py) 的以下实用代码来提取模型的改进指令和输出。",
        "- Let's use these utility functions to extract the improved instruction and response from the lengthy GPT-4 output generated earlier:": "- 让我们使用这些实用函数从之前生成的冗长 GPT-4 输出中提取改进的指令和回复：",
        "- Note that the instruction-refinement is currently only implemented for dataset entries that don't have an `\"input\"` field": "- 请注意，指令细化目前仅针对没有 `\"input\"` 字段的数据集条目实现。",
        "## Improve Responses": "## 改进回复",
        "- In a similar fashion, we can also apply the Reflection-Tuning refinement process specifically to the dataset responses (i.e., \"output\" fields)\n": "- 以类似的方式，我们也可以将 Reflection-Tuning 细化过程专门应用于数据集回复（即“output”字段）。\n",
        "- Below are two small utility functions from the [Reflection-Tuning repository](https://github.com/tianyi-lab/Reflection_Tuning/blob/main/reflection_code/reflect_response.py) to format the inputs to the GPT-4 model for dataset refinement": "- 下面是两个来自 [Reflection-Tuning 仓库](https://github.com/tianyi-lab/Reflection_Tuning/blob/main/reflection_code/reflect_response.py) 的小实用函数，用于为此数据集细化格式化 GPT-4 模型的输入。",
        "- Again, let's apply it to one of the dataset entries to see how it works, generating the improved response:": "- 再次，让我们将其应用于其中一个数据集条目，看看它是如何工作的，生成改进的回复：",
        "- As we can see above, the response includes an analysis of the original response; we can extract the new response using the following utility function from the [Reflection-Tuning repository](https://github.com/tianyi-lab/Reflection_Tuning/blob/main/reflection_code/reflect_response.py)": "- 如上所示，回复包括对原始回复的分析；我们可以使用来自 [Reflection-Tuning 仓库](https://github.com/tianyi-lab/Reflection_Tuning/blob/main/reflection_code/reflect_response.py) 的以下实用函数提取新回复。",
        "## Improving the Dataset": "## 改进数据集",
        "- Now, let's apply the instruction-reflection and response-reflection techniques to the actual dataset\n": "- 现在，让我们将指令反思和回复反思技术应用于实际数据集。\n",
        "- Note: we only apply it to a small data subset here for demo purposes; to apply it to the whole dataset, change\n": "- 注意：为了演示目的，我们这里只将其应用于一个小数据子集；要将其应用于整个数据集，请更改\n",
        "### Reflect instructions": "### 反思指令",
        "- The following code applies the Reflection-Tuning methodology for dataset refinement to the instructions in the original dataset": "- 以下代码将 Reflection-Tuning 数据集细化方法应用于原始数据集中的指令。",
        "- Let's save the new dataset:": "- 让我们保存新数据集：",
        "### Reflect responses": "### 反思回复",
        "- Let's now do the same for the response-reflection:": "- 现在让我们对回复反思做同样的事情：",
        "## Creating Improved Instruction Data": "## 创建改进的指令数据",
        "- Applying the two methodologies above to all 1100 entries in the chapter 7 instruction dataset costs about \\$0.60 (60 cents)\n": "- 将上述两种方法应用于第 7 章指令数据集中的所有 1100 个条目大约需要花费 0.60 美元（60 美分）。\n",
        "- To avoid bloating the GitHub repository with dataset files, the resulting dataset files are available from Google Drive:\n": "- 为了避免 GitHub 仓库因数据集文件而臃肿，可以从 Google Drive 获取结果数据集文件：\n"
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
    translate_notebook('ch07/05_dataset-generation/reflection-gpt4.ipynb', 'ch07/05_dataset-generation/reflection-gpt4_zh.ipynb')
