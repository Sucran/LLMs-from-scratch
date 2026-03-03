# -*- coding: utf-8 -*-
import json
import os

def translate_notebook():
    source_path = 'ch04/01_main-chapter-code/ch04.ipynb'
    target_path = 'ch04/01_main-chapter-code/ch04_zh.ipynb'
    
    print(f"Reading {source_path}...")
    with open(source_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb['cells']
    
    # Helper to safely set cell source
    def set_source(idx, text_list):
        if 0 <= idx < len(cells):
            # Ensure text_list is a list of strings with newlines if needed, 
            # but usually we can just pass a list of strings.
            # If passing a single string, wrap it in a list.
            if isinstance(text_list, str):
                text_list = [text_list]
            cells[idx]['source'] = text_list

    # Translation Mapping based on index
    # Note: Indexing relies on the file structure remaining constant.
    
    # Cell 1: Title
    set_source(1, ["# 第 4 章：从零开始实现 GPT 模型以生成文本"])
    
    # Cell 3: Intro bullet
    set_source(3, ["- 在本章中，我们将实现一个类似 GPT 的大语言模型（LLM）架构；下一章将专注于训练这个 LLM"])
    
    # Cell 5: Section 4.1
    set_source(5, ["&nbsp;\n", "## 4.1 编写 LLM 架构代码"])
    
    # Cell 6: Context
    set_source(6, [
        "- 第 1 章讨论了像 GPT 和 Llama 这样的模型，它们按顺序生成单词，并基于原始 Transformer 架构的解码器部分\n",
        "- 因此，这些 LLM 通常被称为“类解码器”LLM\n",
        "- 与传统深度学习模型相比，LLM 的规模更大，主要是由于参数数量巨大，而不是代码量大\n",
        "- 我们将看到，LLM 架构中有许多重复的元素"
    ])
    
    # Cell 8: More context
    set_source(8, [
        "- 在前面的章节中，为了便于说明，我们使用了较小的嵌入维度来表示输入和输出标记，确保它们能在一页中展示\n",
        "- 在本章中，我们考虑类似于小型 GPT-2 模型的嵌入和模型大小\n",
        "- 我们将专门编写最小的 GPT-2 模型（1.24 亿参数）的架构，正如 Radford 等人在 [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) 中所述（注意：最初的报告将其列为 1.17 亿参数，但这后来在模型权重仓库中得到了修正）\n",
        "- 第 6 章将展示如何将预训练权重加载到我们的实现中，这将与 3.45 亿、7.62 亿和 15.42 亿参数的模型大小兼容"
    ])
    
    # Cell 9: Config intro
    set_source(9, ["- 1.24 亿参数 GPT-2 模型的配置细节包括："])
    
    # Cell 11: Config explanation
    set_source(11, [
        "- 我们使用简短的变量名以避免以后代码行过长\n",
        "- `\"vocab_size\"` 表示词汇表大小为 50,257 个单词，由第 2 章讨论的 BPE 分词器支持\n",
        "- `\"context_length\"` 代表模型的最大输入标记计数，由第 2 章涵盖的位置嵌入启用\n",
        "- `\"emb_dim\"` 是输入标记的嵌入大小，将每个输入标记转换为 768 维向量\n",
        "- `\"n_heads\"` 是第 3 章实现的自注意力机制中的注意力头数\n",
        "- `\"n_layers\"` 是模型中的 Transformer 块数，我们将在后续部分实现\n",
        "- `\"drop_rate\"` 是第 3 章讨论的 Dropout 机制的强度；0.1 意味着在训练期间丢弃 10% 的隐藏单元以减轻过拟合\n",
        "- `\"qkv_bias\"` 决定多头注意力机制（来自第 3 章）中的 `Linear` 层在计算查询（Q）、键（K）和值（V）张量时是否应包含偏置向量；我们将禁用此选项，这是现代 LLM 的标准做法；但是，稍后在第 5 章将 OpenAI 的预训练 GPT-2 权重加载到我们的重新实现中时，我们会重新审视这一点"
    ])

    # NOTE: Since I haven't read the whole file, I will translate the rest based on general knowledge or just translate these first few and append logic for the rest if I can read them or just do a partial translation and then iterate.
    # However, to do a good job, I should probably read the rest of the markdown cells or use a more robust way. 
    # But given the tool usage limits and the user's request, I will try to read the rest of the file first before finalizing the script if possible.
    # Actually, I'll just write the script to print the markdown cells I haven't translated yet so I can see them, OR I can just read the file again.
    
    pass

if __name__ == "__main__":
    translate_notebook()
