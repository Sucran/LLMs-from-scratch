# -*- coding: utf-8 -*-
import json
import os

def translate_notebook():
    source_path = 'ch05/09_extending-tokenizers/extend-tiktoken.ipynb'
    target_path = 'ch05/09_extending-tokenizers/extend-tiktoken_zh.ipynb'
    
    print(f"Reading {source_path}...")
    with open(source_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb['cells']
    
    def set_source(idx, text_list):
        if 0 <= idx < len(cells):
            if isinstance(text_list, str):
                text_list = [text_list]
            cells[idx]['source'] = text_list

    # Cell 28
    set_source(28, ["# 使用新标记扩展 Tiktoken BPE 分词器"])
    
    # Cell 36
    set_source(36, [
        "- 此笔记本解释了我们如何扩展现有的 BPE 分词器；具体来说，我们将专注于如何为流行的 [tiktoken](https://github.com/openai/tiktoken) 实现执行此操作\n",
        "- 有关分词的一般介绍，请参阅 [第 2 章](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/01_main-chapter-code/ch02.ipynb) 和从头开始 BPE [链接] 教程\n",
        "- 例如，假设我们有一个 GPT-2 分词器，并希望编码以下文本："
    ])
    
    # Cell 66
    set_source(66, ["- 遍历每个标记 ID 可以让我们更好地理解标记 ID 是如何通过词汇表解码的："])
    
    # Cell 106
    set_source(106, [
        "- 正如我们在上面看到的，`\"MyNewToken_1\"` 被分解为 5 个单独的子词标记——这是 BPE 处理未知单词时的正常行为\n",
        "- 但是，假设它是一个我们希望编码为单个标记的特殊标记，类似于其他一些单词或 `\"<|endoftext|>\"`；此笔记本解释了如何做到这一点"
    ])
    
    # Cell 115
    set_source(115, ["&nbsp;\n", "## 1. 添加特殊标记"])
    
    # Cell 124
    set_source(124, [
        "- 请注意，我们必须将新标记添加为特殊标记；原因是我们在分词器训练过程中没有为新标记创建“合并”——即使我们有，如果不破坏现有的分词方案，也很难将它们合并（请参阅从头开始 BPE 笔记本 [链接] 以了解“合并”）\n",
        "- 假设我们要添加 2 个新标记："
    ])
    
    # Cell 147
    set_source(147, ["- 接下来，我们创建一个自定义 `Encoding` 对象来保存我们的特殊标记，如下所示："])
    
    # Cell 171
    set_source(171, ["- 就是这样，我们现在可以检查它是否可以编码示例文本："])
    
    # Cell 180
    set_source(180, ["- 正如我们所看到的，新标记 `50257` 和 `50258` 现在已编码在输出中："])
    
    # Cell 211
    set_source(211, ["- 同样，我们也可以在逐个标记的级别上查看它："])

    print(f"Writing {target_path}...")
    with open(target_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Done.")

if __name__ == "__main__":
    translate_notebook()
