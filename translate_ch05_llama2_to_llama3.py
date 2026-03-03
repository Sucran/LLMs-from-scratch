# -*- coding: utf-8 -*-
import json
import os

def translate_notebook():
    source_path = 'ch05/07_gpt_to_llama/converting-llama2-to-llama3.ipynb'
    target_path = 'ch05/07_gpt_to_llama/converting-llama2-to-llama3_zh.ipynb'
    
    print(f"Reading {source_path}...")
    with open(source_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb['cells']
    
    def set_source(idx, text_list):
        if 0 <= idx < len(cells):
            if isinstance(text_list, str):
                text_list = [text_list]
            cells[idx]['source'] = text_list

    # Cell 32
    set_source(32, ["# 将 Llama 2 转换为 Llama 3"])
    
    # Cell 41
    set_source(41, [
        "- 在本笔记本中，我们将我们在 [converting-gpt-to-llama2.ipynb](converting-gpt-to-llama2.ipynb) 笔记本中实现的 Llama 2 架构转换为 Llama 3（以及 Llama 3.1 和 Llama 3.2）\n",
        "- Llama 3 架构与 Llama 2 几乎相同；唯一的主要区别是词汇表大小，以及 8B 模型使用了分组查询注意力（GQA），而 7B Llama 2 模型使用了多头注意力（MHA）\n",
        "- 但是，请注意，较大的 Llama 2 模型（如 70B 变体）也使用了 GQA，所以如果你已经实现了 Llama 2 70B，那么唯一的区别就是词汇表大小"
    ])
    
    # Cell 66
    set_source(66, ["- 本笔记本中使用的包："])
    
    # Cell 110
    set_source(110, ["&nbsp;\n", "## 1. 更新 Llama 2 架构"])
    
    # Cell 121
    set_source(121, [
        "- 在本节中，我们复制 [converting-gpt-to-llama2.ipynb](converting-gpt-to-llama2.ipynb) 笔记本中的代码，并进行必要的调整以实现 Llama 3\n",
        "- 唯一的主要变化是我们用 GroupedQueryAttention 替换了 MultiHeadAttention（这是我们在第 3 章中实现的多头注意力的更高效变体）"
    ])
    
    # Cell 132
    set_source(132, ["&nbsp;\n", "## 1.1 支持类（与 Llama 2 相同）"])
    
    # Cell 143
    set_source(143, ["- 以下类与 Llama 2 中的完全相同，因此我们可以直接复制它们："])
    
    # Cell 305
    set_source(305, ["&nbsp;\n", "## 1.2 分组查询注意力"])
    
    # Cell 458
    set_source(458, [
        "- 在本节中，我们将多头注意力 (MHA) 替换为称为分组查询注意力 (GQA) 的替代机制\n",
        "- 简而言之，可以将 GQA 视为 MHA 的计算和参数更高效的版本\n",
        "- 在 GQA 中，我们通过在多个注意力头之间共享键和值投影来减少它们的数量\n",
        "- 每个注意力头仍然有其唯一的查询，但这些查询关注同一组键和值\n",
        "- 下面是具有 2 个键值组 (kv-groups) 的 GQA 说明：\n",
        "\n",
        "<img src=\"https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gpt-to-llama/grouped-query-attention.webp\" width=\"500px\">"
    ])
    
    # Cell 474
    set_source(474, [
        "- GQA 背后的主要思想是减少关注键值对的唯一查询组的数量，从而在不显着降低建模性能的情况下减少某些矩阵乘法的大小和 MHA 中的参数数量\n",
        "- GQA 代码与 MHA 非常相似（我在下面通过“NEW”部分突出显示了更改）\n",
        "- 简而言之，GQA 中的主要更改是每个查询组需要重复以匹配与其关联的头数，如下所示实现"
    ])
    
    # Cell 484
    set_source(484, [
        "- **我们还稍微重新设计了注意力类，使其通过其 forward 方法接收掩码，而不是将其存储并作为 `self.mask` 访问。这使我们能够动态构建掩码以减少内存使用。预示一下原因：Llama 3.1 可以处理多达 128k 个标记的序列，预先计算 128k × 128k 的因果掩码将非常占用内存，因此除非绝对必要，否则我们避免这样做。**"
    ])
    
    # Cell 606
    set_source(606, ["- 为了说明 GQA 相对于 MHA 的参数节省，请考虑以下来自 GPT 和 Llama 2 代码的多头注意力示例："])
    
    # Cell 663
    set_source(663, ["- 现在，如果我们改用分组查询注意力，使用 8 个 kv 组（这就是 Llama 3 8B 使用的数量），我们可以看到键和值矩阵的行数减少了 4 倍（因为 32 个注意力头除以 8 个 kv 组等于 4）"])
    
    # Cell 710
    set_source(710, [
        "- 顺便说一句，要使 GroupedQueryAttention 等效于标准多头注意力，您可以将查询组的数量 (`num_kv_groups`) 设置为等于头数 (`num_heads`)\n",
        "- 最后，让我们比较下面的参数数量："
    ])
    
    # Cell 766
    set_source(766, ["&nbsp;\n", "## 1.4 更新 TransformerBlock 模块"])
    
    # Cell 778
    set_source(778, [
        "- 接下来，我们更新 `TransformerBlock`\n",
        "- 在这里，我们只需将 `MultiHeadAttention` 替换为 `GroupedQueryAttention` 并添加新的 RoPE 设置\n",
        "- 此外，我们还修改了 `forward` 方法，使其接收 `mask`、`cos` 和 `sin`；由于每个 Transformer 块的这些值都相同，我们只需要计算一次，然后就可以重用它们"
    ])
    
    # Cell 833
    set_source(833, ["&nbsp;\n", "## 1.5 定义模型类"])
    
    # Cell 844
    set_source(844, [
        "- 在设置模型类时，我们在技术上不需要做太多事情；我们只需将名称更新为 `Llama3Model`\n",
        "- 但是，由于我们现在将 `mask`、`cos` 和 `sin` 传递给 Transformer 块，我们还必须在这里添加它们"
    ])
    
    # Cell 906
    set_source(906, ["&nbsp;\n", "## 2. 初始化模型"])
    
    # Cell 916
    set_source(916, ["- 现在我们可以定义一个 Llama 3 配置文件（Llama 2 配置文件用于比较）"])
    
    # Cell 969
    set_source(969, [
        "- 使用这些设置，我们现在可以初始化一个 Llama 3 8B 模型\n",
        "- 请注意，这需要约 34 GB 的内存（相比之下，Llama 2 7B 需要约 26 GB 的内存）"
    ])
    
    # Cell 991
    set_source(991, ["- 现在让我们计算可训练参数的数量："])
    
    # Cell 1026
    set_source(1026, [
        "- 如上所示，该模型包含 80 亿个参数\n",
        "- 此外，我们可以使用下面的代码计算此模型的内存需求："
    ])
    
    # Cell 1087
    set_source(1087, ["- 最后，如果适用，我们还可以将模型传输到 NVIDIA 或 Apple Silicon GPU："])
    
    # Cell 1116
    set_source(1116, ["&nbsp;\n", "## 3. 加载分词器"])
    
    # Cell 1127
    set_source(1127, [
        "- 在本节中，我们将加载模型的分词器\n",
        "- Llama 2 使用 Google 的 [SentencePiece](https://github.com/google/sentencepiece) 分词器而不是 OpenAI 的基于 [Tiktoken](https://github.com/openai/tiktoken) 库的 BPE 分词器\n",
        "- 然而，Llama 3 恢复使用 Tiktoken 的 BPE 分词器；具体来说，它使用具有扩展词汇表的 GPT-4 分词器\n",
        "- 您可以在他们的官方 Llama 3 存储库中找到 Meta AI 的原始 Tiktoken 改编版 [这里](https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py)\n",
        "- 下面，我重写了分词器代码，使其对于本笔记本更具可读性和最小化（但行为应该相似）"
    ])
    
    # Cell 1200
    set_source(1200, [
        "- Meta AI 在 Hugging Face Hub 上分享了原始 Llama 3 模型权重和分词器词汇表\n",
        "- 我们将首先从 Hub 下载分词器词汇表并将其加载到上面的代码中"
    ])
    
    # Cell 1211
    set_source(1211, [
        "- 请注意，Meta AI 要求您在下载文件之前接受 Llama 3 许可条款；为此，您必须创建一个 Hugging Face Hub 帐户并访问 [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) 存储库以接受条款\n",
        "- 接下来，您需要创建一个访问令牌；要生成具有 READ 权限的访问令牌，请单击右上角的个人资料图片，然后单击“Settings”\n",
        "\n",
        "\n",
        "<img src=\"https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gpt-to-llama/settings.webp?1\" width=\"300px\">\n",
        "\n",
        "- 然后，创建并复制访问令牌，以便您可以将其复制并粘贴到下一个代码单元中\n",
        "\n",
        "<img src=\"https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gpt-to-llama/access-token.webp?1\" width=\"600px\">"
    ])
    
    # Cell 1252
    set_source(1252, ["- 通过访问令牌登录后（这是验证我们要接受 Llama 3 许可条款所必需的），我们现在可以下载分词器词汇表："])
    
    # Cell 1299
    set_source(1299, [
        "- 请注意，为了使用 Llama 3 文件，我们可能需要 `blobfile` 包，该包用于处理存储在云存储解决方案（如 Google Cloud Storage (GCS)、Azure Blob Storage 或 Amazon S3）中的数据集或模型\n",
        "- 您可以通过取消注释并执行下面的 `pip` 命令来安装此依赖项"
    ])
    
    # Cell 1334
    set_source(1334, ["- 我们现在可以使用 `generate` 函数让 Llama 3 模型生成新文本："])
    
    # Cell 1392
    set_source(1392, [
        "- 当然，正如我们在上面看到的，文本是荒谬的，因为我们还没有训练 Llama 3 模型\n",
        "- 在下一节中，我们不是自己训练它（这将花费数万到数十万美元），而是从 Meta AI 加载预训练权重"
    ])
    
    # Cell 1403
    set_source(1403, ["&nbsp;\n", "## 4. 加载预训练权重"])
    
    # Cell 1414
    set_source(1414, [
        "- 我们在下面加载 [\"meta-llama/Meta-Llama-3-8B\"](https://huggingface.co/meta-llama/Meta-Llama-3-8B) 基础模型，这是一个在微调之前的简单文本补全模型\n",
        "- 或者，您可以通过相应地修改下一个代码单元中的字符串来加载指令微调和对齐的 [\"meta-llama/Meta-Llama-3-8B-Instruct\"](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) 模型\n",
        "- 权重文件总共约 16 GB 大"
    ])
    
    # Cell 1557
    set_source(1557, ["- `weights` 包含以下张量（为简单起见仅显示前 15 个）："])
    
    # Cell 1608
    set_source(1608, ["- 以下函数仿照 [第 5 章](../01_main-chapter-code/ch05.ipynb) 中的 `load_weights_into_gpt` 函数，将预训练权重加载到我们的 Llama 3 模型中："])
    
    # Cell 1709
    set_source(1709, ["- 接下来，我们准备使用该模型进行文本生成"])
    
    # Cell 1755
    set_source(1755, ["&nbsp;\n", "## 5. 使用指令微调模型"])
    
    # Cell 1766
    set_source(1766, ["- 如前所述，上面我们使用了预训练的基础模型；如果您想使用能够遵循指令的模型，请改用 `\"meta-llama/Llama-3-8B-Instruct\"` 模型，如下所示"])
    
    # Cell 1931
    set_source(1931, [
        "- 请注意，Llama 3 模型最好使用微调期间使用的正确提示模板（如第 7 章所讨论）\n",
        "- 下面是基于 Meta AI 的 Llama 3 特定 [ChatFormat 代码](https://github.com/meta-llama/llama3/blob/11817d47e1ba7a4959b025eb1ca308572e0e3963/llama/tokenizer.py#L202) 的分词器包装类，用于构建提示模板"
    ])
    
    # Cell 1990
    set_source(1990, ["- 用法如下："])
    
    # Cell 2055
    set_source(2055, ["- 现在让我们看看 Llama 3 指令模型的实际应用："])
    
    # Cell 2121
    set_source(2121, ["&nbsp;\n", "## 6. Llama 3.1 8B"])
    
    # Cell 2131
    set_source(2131, [
        "- 在最初的 Llama 3 发布几个月后，Meta AI 推出了他们的 Llama 3.1 模型套件（详情请参阅官方 [Introducing Llama 3.1: Our most capable models to date](https://ai.meta.com/blog/meta-llama-3-1/) 公告博客文章）\n",
        "- 方便的是，我们可以重用上面之前的 Llama 3 代码来实现 Llama 3.1 8B\n",
        "\n",
        "<img src=\"https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gpt-to-llama/llama3-to-llama31.webp\" width=\"700px\">\n",
        "\n",
        "- 架构完全相同，唯一的变化是 RoPE 频率的重新缩放，如下面的配置文件所示"
    ])
    
    # Cell 2189
    set_source(2189, [
        "- 正如我们之前在代码中看到的，RoPE 方法使用正弦函数（正弦和余弦）将位置信息直接嵌入到注意力机制中\n",
        "- 在 Llama 3.1 中，通过附加配置，我们对逆频率计算引入了额外的调整\n",
        "- 这些调整会影响不同频率分量对位置嵌入的贡献（详细解释是另一个话题）\n",
        "- 让我们在实践中尝试 Llama 3.1 模型；首先，我们清除旧模型以释放一些 GPU 内存"
    ])
    
    # Cell 2220
    set_source(2220, [
        "- 接下来，我们下载分词器\n",
        "- 请注意，由于 Llama 3.1 系列与 Llama 3 系列不同，您必须转到 [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) 存储库并确认许可条款，以便您的 Hugging Face 访问令牌可以用于下载\n",
        "- 提示：为简单起见，我们下面只加载基础模型，但也有一个指令微调版本，您可以通过将 `\"meta-llama/Llama-3.1-8B\"` 替换为 `\"meta-llama/Llama-3.1-8B-Instruct\"` 来使用"
    ])
    
    # Cell 2463
    set_source(2463, ["&nbsp;\n", "## 7. Llama 3.2 1B"])
    
    # Cell 2472
    set_source(2472, [
        "- 截至撰写本文时，Meta AI 的最新模型是 [此处](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) 宣布的 Llama 3.2 模型\n",
        "- Llama 3.2 文本模型的代码与 Llama 3.1 类似，只是模型尺寸缩小了（有 1B 和 3B 版本）\n",
        "- 另一个效率调整是他们加回了权重绑定（这是最初在 GPT-2 架构中使用的概念）；在这里，他们在输入（标记）嵌入层和输出层中重用相同的权重参数值\n",
        "- Llama 3.2 1B 的小模型尺寸非常方便，因为它甚至可以在许多移动设备上运行\n",
        "- Llama 3.1 8B 和 Llama 3.2 1B 之间的架构差异如下图所示"
    ])
    
    # Cell 2496
    set_source(2496, [
        "<img src=\"https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gpt-to-llama/llama31-to-llama32.webp?1\" width=\"700px\">\n",
        "\n",
        "- 正如我们从上图所看到的，Llama 3.1 8B 和 Llama 3.2 1B 架构之间的主要区别在于各自的大小\n",
        "- 一个小的额外变化是增加了 RoPE 重新缩放因子，这反映在下面的配置文件中"
    ])
    
    # Cell 2554
    set_source(2554, [
        "- 下面，我们可以重用 Llama 3.1 8B 部分的代码来加载 Llama 3.2 1B 模型\n",
        "- 同样，由于 Llama 3.2 系列与 Llama 3.1 系列不同，您必须转到 [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) 存储库并确认许可条款，以便您的 Hugging Face 访问令牌可以用于下载\n",
        "- 提示：为简单起见，我们下面只加载基础模型，但也有一个指令微调版本，您可以通过将 `\"meta-llama/Llama-3.2-1B\"` 替换为 `\"meta-llama/Llama-3.2-1B-Instruct\"` 来使用"
    ])
    
    # Cell 2650
    set_source(2650, ["- 或者，我们可以使用更健壮的函数，该函数基于内存中的共享数据指针来考虑权重绑定，如 [#822](https://github.com/rasbt/LLMs-from-scratch/issues/822) 中所建议的那样："])
    
    # Cell 2820
    set_source(2820, ["&nbsp;\n", "# 下一步是什么？"])
    
    # Cell 2828
    set_source(2828, [
        "- 本笔记本总结了从 GPT 到 Llama 3.2 的转换\n",
        "- 如果您对更紧凑、独立的笔记本感兴趣，其中仅包含 Llama 3.2 代码，请查看 [standalone-llama32.ipynb](standalone-llama32.ipynb) 笔记本"
    ])

    print(f"Writing {target_path}...")
    with open(target_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Done.")

if __name__ == "__main__":
    translate_notebook()
