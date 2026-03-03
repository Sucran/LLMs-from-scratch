# -*- coding: utf-8 -*-
import json
import os

def translate_notebook(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    translations = {
        "Supplementary code for the <a href=\"http://mng.bz/orYv\">Build a Large Language Model From Scratch</a> book by <a href=\"https://sebastianraschka.com\">Sebastian Raschka</a><br>\n": "《<a href=\"http://mng.bz/orYv\">从头开始构建大型语言模型</a>》一书的补充代码，作者 <a href=\"https://sebastianraschka.com\">Sebastian Raschka</a><br>\n",
        "<br>Code repository: <a href=\"https://github.com/rasbt/LLMs-from-scratch\">https://github.com/rasbt/LLMs-from-scratch</a>\n": "<br>代码仓库：<a href=\"https://github.com/rasbt/LLMs-from-scratch\">https://github.com/rasbt/LLMs-from-scratch</a>\n",
        "# Chapter 7 Exercise solutions": "# 第 7 章 练习解答",
        "## Exercise 7.1: Changing prompt styles": "## 练习 7.1：更改提示风格",
        "## Exercise 7.2: Instruction and input masking": "## 练习 7.2：指令和输入屏蔽",
        "## Exercise 7.3: Finetuning on the original Alpaca dataset": "## 练习 7.3：在原始 Alpaca 数据集上微调",
        "## Exercise 7.4: Parameter-efficient finetuning with LoRA": "## 练习 7.4：使用 LoRA 进行参数高效微调",
        "#### Tip: Considering special tokens": "#### 提示：考虑特殊标记",
        "Suppose we have the following data entry:\n": "假设我们有以下数据条目：\n",
        "In the main chapter, we formatted it according to the Alpaca-style prompt template:\n": "在正文章节中，我们根据 Alpaca 风格的提示模板对其进行了格式化：\n",
        "In this exercise, we now use the Phi-3 prompt template instead, which formats the data entry as follows:\n": "在这个练习中，我们现在改用 Phi-3 提示模板，它将数据条目格式化如下：\n",
        "Note that this prompt template is substantially shorter, which reduces the runtime and hardware requirements for finetuning the LLM and generating text since the input prompts are shorter.\n": "请注意，此提示模板要短得多，这减少了微调 LLM 和生成文本的运行时间和硬件要求，因为输入提示更短。\n",
        "To make this change, we update the `format_input` function as follows:": "为了进行此更改，我们更新 `format_input` 函数，如下所示：",
        "Let's make sure that it works as intended by applying it to two input samples, one with and one without content in the `'input'` field:": "让我们通过将其应用于两个输入样本（一个在 `'input'` 字段中有内容，一个没有）来确保它按预期工作：",
        "Next, we also update the `InstructionDataset` class to use the <|assistant|> prompt template for the response:": "接下来，我们还更新 `InstructionDataset` 类以使用 <|assistant|> 提示模板进行响应：",
        "Lastly, we also have to update the way we extract the generated response when we collect the test set responses:": "最后，我们还必须更新我们在收集测试集响应时提取生成响应的方式：",
        "For your convenience, the exercise solution is implemented in the [exercise_experiments.py](exercise_experiments.py) script, which you can run as follows:": "为了您的方便，练习解答已在 [exercise_experiments.py](exercise_experiments.py) 脚本中实现，您可以按如下方式运行：",
        "For comparison, you can run the original chapter 7 finetuning code via `python exercise_experiments.py --exercise_solution baseline`. \n": "为了比较，您可以通过 `python exercise_experiments.py --exercise_solution baseline` 运行原始第 7 章微调代码。\n",
        "Note that on an Nvidia L4 GPU, the code above, using the Phi-3 prompt template, takes 1.5 min to run. In comparison, the Alpaca-style template takes 1.80 minutes to run. So, the Phi-3 template is approximately 17% faster since it results in shorter model inputs. \n": "请注意，在 Nvidia L4 GPU 上，上面使用 Phi-3 提示模板的代码需要 1.5 分钟运行。相比之下，Alpaca 风格的模板需要 1.80 分钟运行。因此，Phi-3 模板快了大约 17%，因为它导致更短的模型输入。\n",
        "Let's take a look at some of the responses to make sure they have been formatted correctly:\n": "让我们看一些响应，以确保它们的格式正确：\n",
        "We can evaluate the performance using the Ollama Llama 3 method, which is for your convenience, also implemented in the `python exercise_experiments.py` script, which we can run as follows:\n": "我们可以使用 Ollama Llama 3 方法评估性能，为了您的方便，该方法也在 `python exercise_experiments.py` 脚本中实现，我们可以按如下方式运行：\n",
        "The score is close to 50, which is in the same ballpark as the score we previously achieved with the Alpaca-style prompts.\n": "分数接近 50，这与我们之前使用 Alpaca 风格提示获得的分数大致相同。\n",
        "There is no inherent advantage or rationale why the Phi prompt-style should be better, but it can be more concise and efficient, except for the caveat mentioned in the *Tip* section below.": "Phi 提示风格并没有固有的优势或理由更好，但它可能更简洁高效，除了下面 *提示* 部分提到的注意事项。",
        "- Note that the Phi-3 prompt template contains special tokens such as `<|user|>` and `<|assistant|>`, which can be suboptimal for the GPT-2 tokenizer\n": "- 请注意，Phi-3 提示模板包含特殊标记，例如 `<|user|>` 和 `<|assistant|>`，这对于 GPT-2 分词器可能不是最佳的\n",
        "- While the GPT-2 tokenizer recognizes `<|endoftext|>` as a special token (encoded into token ID 50256), it is inefficient at handling other special tokens, such as the aforementioned ones\n": "- 虽然 GPT-2 分词器将 `<|endoftext|>` 识别为特殊标记（编码为标记 ID 50256），但它在处理其他特殊标记（例如上述标记）时效率低下\n",
        "- For instance, `<|user|>` is encoded into 5 individual token IDs (27, 91, 7220, 91, 29), which is very inefficient\n": "- 例如，`<|user|>` 被编码为 5 个单独的标记 ID (27, 91, 7220, 91, 29)，这非常低效\n",
        "- We could add `<|user|>` as a new special token in `tiktoken` via the `allowed_special` argument, but please keep in mind that the GPT-2 vocabulary would not be able to handle it without additional modification\n": "- 我们可以通过 `allowed_special` 参数在 `tiktoken` 中将 `<|user|>` 添加为新的特殊标记，但请记住，如果没有额外的修改，GPT-2 词汇表将无法处理它\n",
        "- If you are curious about how a tokenizer and LLM can be extended to handle special tokens, please see the [extend-tiktoken.ipynb](../../ch05/09_extending-tokenizers/extend-tiktoken.ipynb) bonus materials (note that this is not required here but is just an interesting/bonus consideration for curious readers)\n": "- 如果您对如何扩展分词器和 LLM 以处理特殊标记感到好奇，请参阅 [extend-tiktoken.ipynb](../../ch05/09_extending-tokenizers/extend-tiktoken.ipynb) 奖励材料（请注意，这里不需要这样做，这只是供好奇读者的有趣/奖励考虑）\n",
        "- Furthermore, we can hypothesize that models that support these special tokens of a prompt template via their vocabulary may perform more efficiently and better overall": "- 此外，我们可以假设通过其词汇表支持提示模板的这些特殊标记的模型可能会表现得更有效率，总体上也更好",
        "To mask out the instructions as shown in the following figure, we need to make slight modifications to the `InstructionDataset` class and `custom_collate_fn`.\n": "为了屏蔽指令，如下图所示，我们需要对 `InstructionDataset` 类和 `custom_collate_fn` 进行轻微修改。\n",
        "We can modify the `InstructionDataset` class to collect the lengths of the instructions, which we will use in the collate function to locate the instruction content positions in the targets when we code the collate function, as follows:": "我们可以修改 `InstructionDataset` 类以收集指令的长度，我们在编写 collate 函数时将在 collate 函数中使用这些长度来定位目标中的指令内容位置，如下所示：",
        "Next, we update the `custom_collate_fn` where each `batch` is now a tuple containing `(instruction_length, item)` instead of just `item` due to the changes in the `InstructionDataset` dataset. In addition, we now mask the corresponding instruction tokens in the target ID list.": "接下来，我们更新 `custom_collate_fn`，由于 `InstructionDataset` 数据集的更改，现在的每个 `batch` 都是包含 `(instruction_length, item)` 的元组，而不仅仅是 `item`。此外，我们现在屏蔽目标 ID 列表中的相应指令标记。",
        "Let's try it out on some sample data below:": "让我们在下面的一些样本数据上试用一下：",
        "As we can see based on the `targets` tensor, both the instruction and padding tokens are now masked using the -100 placeholder tokens. \n": "正如我们根据 `targets` 张量所看到的，指令和填充标记现在都使用 -100 占位符标记进行了屏蔽。\n",
        "Let's decode the inputs just to make sure that they look correct:": "让我们解码输入以确保它们看起来正确：",
        "Next, let's decode the non-masked target token IDS:": "接下来，让我们解码未屏蔽的目标标记 ID：",
        "As shown above, the non-masked target tokens exclude the `\"Instruction\"` and `\"Input\"` fields, as intended. Now, we can run the modified code to see how well the LLM performs when finetuned using this masking strategy.\n": "如上所示，未屏蔽的目标标记排除了 `\"Instruction\"` 和 `\"Input\"` 字段，正如预期的那样。现在，我们可以运行修改后的代码，看看使用这种屏蔽策略进行微调时 LLM 的表现如何。\n",
        "For your convenience, you can use the `exercise_experiments.py` code to run a comparison as follows:": "为了您的方便，您可以使用 `exercise_experiments.py` 代码运行比较，如下所示：",
        "Next, let's evaluate the performance of the resulting LLM:\n": "接下来，让我们评估生成的 LLM 的性能：\n",
        "As we can see based on the scores, the instruction masking does perform slightly worse, which is consistent with the observation in the \"Instruction Tuning With Loss Over Instructions\" paper (https://arxiv.org/abs/2405.14394)": "正如我们根据分数所看到的，指令屏蔽的表现确实稍差，这与“Instruction Tuning With Loss Over Instructions”论文 (https://arxiv.org/abs/2405.14394) 中的观察结果一致",
        "To finetune the model on the original Stanford Alpaca dataset ([https://github.com/tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)), you just need to change the file URL from\n": "要在原始 Stanford Alpaca 数据集 ([https://github.com/tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)) 上微调模型，您只需要将文件 URL 从\n",
        "to\n": "更改为\n",
        "Note that the dataset contains 52k entries (50x more than in chapter 7), and the entries are longer than the ones we worked with in chapter 7.\n": "请注意，该数据集包含 5.2 万个条目（比第 7 章多 50 倍），并且条目比我们在第 7 章中处理的条目更长。\n",
        "Thus, it's highly recommended that the training be run on a GPU.\n": "因此，强烈建议在 GPU 上运行训练。\n",
        "If you encounter out-of-memory errors, consider reducing the batch size from 8 to 4, 2, or 1. In addition to lowering the batch size, you may also want to consider lowering the `allowed_max_length` from 1024 to 512 or 256.": "如果遇到内存不足错误，请考虑将批大小从 8 减少到 4、2 或 1。除了降低批大小外，您可能还希望考虑将 `allowed_max_length` 从 1024 降低到 512 或 256。",
        "For your convenience, you can use the `exercise_experiments.py` code to finetune the model on the 52k Alpaca dataset with a batch size of 4 and an `allowed_max_length` of 512 as follows:": "为了您的方便，您可以使用 `exercise_experiments.py` 代码在 5.2 万 Alpaca 数据集上微调模型，批大小为 4，`allowed_max_length` 为 512，如下所示：",
        "Below are a few examples from the Alpaca dataset, including the generated model responses:": "以下是 Alpaca 数据集的一些示例，包括生成的模型响应：",
        "Finally, we can evaluate the finetuned LLM using the [ollama_evaluate.py](ollama_evaluate.py) utility function:\n": "最后，我们可以使用 [ollama_evaluate.py](ollama_evaluate.py) 实用函数评估微调后的 LLM：\n",
        "The score is slightly lower than the score we obtained on the dataset we used in this chapter. However, note that the Alpaca test set contains more diverse and partly more challenging instructions than the dataset we used in the main chapter.": "分数略低于我们在本章中使用的数据集上获得的分数。但是，请注意，Alpaca 测试集包含比我们在正文章节中使用的数据集更多样化且部分更具挑战性的指令。",
        "To instruction finetune the model using LoRA, use the relevant classes and functions from appendix E:\n": "要使用 LoRA 对模型进行指令微调，请使用附录 E 中的相关类和函数：\n",
        "Next, add the following lines of code below the model loading code in section 7.5:\n": "接下来，在第 7.5 节中的模型加载代码下方添加以下代码行：\n",
        "For your convenience, you can use the `exercise_experiments.py` code to finetune the model, using LoRA with rank 16 and alpa 16, as follows:": "为了您的方便，您可以使用 `exercise_experiments.py` 代码来微调模型，使用秩为 16 和 alpha 为 16 的 LoRA，如下所示：",
        "For comparison, you can run the original chapter 7 finetuning code via `python exercise_experiments.py --exercise_solution baseline`. \n": "为了比较，您可以通过 `python exercise_experiments.py --exercise_solution baseline` 运行原始第 7 章微调代码。\n",
        "Note that on an Nvidia L4 GPU, the code above, using LoRA, takes 1.30 min to run. In comparison, the baseline takes 1.80 minutes to run. So, LoRA is approximately 28% faster.\n": "请注意，在 Nvidia L4 GPU 上，上面使用 LoRA 的代码需要 1.30 分钟运行。相比之下，基线需要 1.80 分钟运行。因此，LoRA 快了大约 28%。\n",
        "The score is around 50, which is in the same ballpark as the original model.": "分数约为 50，与原始模型大致相同。"
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
        '/Users/richard/Git/LLMs-from-scratch/ch07/01_main-chapter-code/exercise-solutions.ipynb',
        '/Users/richard/Git/LLMs-from-scratch/ch07/01_main-chapter-code/exercise-solutions_zh.ipynb'
    )
