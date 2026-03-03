# -*- coding: utf-8 -*-
import json
import os

def translate_notebook():
    source_path = 'ch05/01_main-chapter-code/exercise-solutions.ipynb'
    target_path = 'ch05/01_main-chapter-code/exercise-solutions_zh.ipynb'
    
    print(f"Reading {source_path}...")
    with open(source_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb['cells']
    
    def set_source(idx, text_list):
        if 0 <= idx < len(cells):
            if isinstance(text_list, str):
                text_list = [text_list]
            cells[idx]['source'] = text_list

    # Cell 1
    set_source(1, ["# 第 5 章 练习答案"])
    
    # Cell 66
    set_source(66, ["&nbsp;\n", "## 练习 5.1：温度缩放的 softmax 分数和采样概率"])
    
    # Cell 74
    set_source(74, [
        "- 我们可以使用本节中定义的 `print_sampled_tokens` 函数打印单词“pizza”被采样的次数\n",
        "- 让我们从第 5.3.1 节中定义的代码开始\n",
        "\n",
        "- 如果温度为 0 或 0.1，则采样 0 次；如果温度缩放至 5，则采样 32 次。估计概率为 32/1000 * 100% = 3.2%\n",
        "\n",
        "- 实际概率为 4.3%，包含在重新缩放的 softmax 概率张量 (`scaled_probas[2][6]`) 中"
    ])
    
    # Cell 87
    set_source(87, ["- 下面是一个使用第 5 章代码的独立示例："])
    
    # Cell 138
    set_source(138, ["- 现在，我们可以遍历 `scaled_probas` 并打印每种情况下的采样频率："])
    
    # Cell 199
    set_source(199, [
        "- 请注意，当采样单词“pizza”时，采样提供了实际概率的近似值\n",
        "- 例如，如果采样 32/1000 次，则估计概率为 3.2%\n",
        "- 要获得实际概率，我们可以通过访问 `scaled_probas` 中的相应条目直接检查概率\n",
        "\n",
        "- 由于“pizza”是词汇表中的第 7 个条目，对于温度 5，我们按如下方式获得它："
    ])
    
    # Cell 235
    set_source(235, ["如果将温度设置为 5，则采样单词“pizza”的概率为 4.3%"])
    
    # Cell 244
    set_source(244, ["&nbsp;\n", "## 练习 5.2：不同的温度和 top-k 设置"])
    
    # Cell 252
    set_source(252, [
        "- 必须根据单个 LLM 调整温度和 top-k 设置（这是一种试错过程，直到生成所需的输出）\n",
        "- 但是，期望的结果也因具体应用而异\n",
        "  - 较低的 top-k 和温度会导致较少的随机结果，这在创建教育内容、技术写作或问答、数据分析、代码生成等时是所需的\n",
        "  - 较高的 top-k 和温度会导致更多样化和随机的输出，这对于头脑风暴任务、创意写作等更为理想"
    ])
    
    # Cell 264
    set_source(264, ["&nbsp;\n", "## 练习 5.3：解码函数中的确定性行为"])
    
    # Cell 272
    set_source(272, [
        "有多种方法可以使用 `generate` 函数强制执行确定性行为：\n",
        "\n",
        "1. 设置 `temperature=0.0`;\n",
        "2. 设置 `top_k=1`."
    ])
    
    # Cell 283
    set_source(283, ["下面是一个使用第 5 章代码的独立示例："])
    
    # Cell 393
    set_source(393, ["- 请注意，重新执行上一个代码单元将生成完全相同的文本："])
    
    # Cell 432
    set_source(432, ["&nbsp;\n", "## 练习 5.4：继续预训练"])
    
    # Cell 440
    set_source(440, [
        "- 如果我们仍然在第 5 章首次训练模型的 Python 会话中，为了继续预训练一个 epoch，我们只需要加载我们在主章节中保存的模型和优化器，然后再次调用 `train_model_simple` 函数\n",
        "\n",
        "- 在这个新的代码环境中使其可重现需要多几个步骤\n",
        "- 首先，我们加载分词器、模型和优化器："
    ])
    
    # Cell 487
    set_source(487, ["- 接下来，我们初始化数据加载器："])
    
    # Cell 570
    set_source(570, ["- 最后，我们使用 `train_model_simple` 函数来训练模型："])
    
    # Cell 606
    set_source(606, ["&nbsp;\n", "## 练习 5.5：预训练模型的训练集和验证集损失"])
    
    # Cell 613
    set_source(613, [
        "- 我们可以使用以下代码来计算 GPT 模型的训练集和验证集损失：\n",
        "\n",
        "```python\n",
        "train_loss = calc_loss_loader(train_loader, gpt, device)\n",
        "val_loss = calc_loss_loader(val_loader, gpt, device)\n",
        "```\n",
        "\n",
        "- 124M 参数的生成损失如下：\n",
        "\n",
        "```\n",
        "Training loss: 3.754748503367106\n",
        "Validation loss: 3.559617757797241\n",
        "```\n",
        "\n",
        "- 主要观察结果是训练集和验证集的性能大致相同\n",
        "- 这可能有多种解释：\n",
        "\n",
        "1. 当 OpenAI 训练 GPT-2 时，The Verdict 不是预训练数据集的一部分。因此，模型没有明确地过拟合训练集，并且在 The Verdict 的训练集和验证集部分表现同样出色。（验证集损失略低于训练集损失，这在深度学习中是不寻常的。但是，这可能是由于随机噪声，因为数据集相对较小。在实践中，如果没有过拟合，预计训练集和验证集的性能大致相同）。\n",
        "\n",
        "2. The Verdict 是 GPT-2 训练数据集的一部分。在这种情况下，我们无法判断模型是否过拟合训练数据，因为验证集也将用于训练。为了评估过拟合的程度，我们需要一个在 OpenAI 完成 GPT-2 训练后生成的新数据集，以确保它不可能是预训练的一部分。"
    ])
    
    # Cell 640
    set_source(640, ["下面的代码是此新笔记本的可重现独立示例。"])
    
    # Cell 824
    set_source(824, ["我们也可以对最大的 GPT-2 模型重复此操作，但不要忘记更新上下文长度："])
    
    # Cell 883
    set_source(883, ["&nbsp;\n", "## 练习 5.6：尝试更大的模型"])
    
    # Cell 890
    set_source(890, [
        "- 在主要章节中，我们实验了最小的 GPT-2 模型，它只有 124M 参数\n",
        "- 原因是尽可能保持资源需求低\n",
        "- 但是，您可以通过最少的代码更改轻松地尝试更大的模型\n",
        "- 例如，在第 5 章中加载 1558M 而不是 124M 模型，我们唯一需要更改的 2 行代码是\n",
        "\n",
        "```python\n",
        "settings, params = download_and_load_gpt2(model_size=\"124M\", models_dir=\"gpt2\")\n",
        "model_name = \"gpt2-small (124M)\"\n",
        "```\n",
        "\n",
        "- 更新后的代码变为\n",
        "\n",
        "\n",
        "```python\n",
        "settings, params = download_and_load_gpt2(model_size=\"1558M\", models_dir=\"gpt2\")\n",
        "model_name = \"gpt2-xl (1558M)\"\n",
        "```"
    ])

    print(f"Writing {target_path}...")
    with open(target_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Done.")

if __name__ == "__main__":
    translate_notebook()
