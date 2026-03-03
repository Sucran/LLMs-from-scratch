# -*- coding: utf-8 -*-
import json
import os

def translate_notebook():
    source_path = 'ch04/01_main-chapter-code/exercise-solutions.ipynb'
    target_path = 'ch04/01_main-chapter-code/exercise-solutions_zh.ipynb'
    
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
    set_source(1, ["# 第 4 章 练习答案"])
    
    # Cell 4
    set_source(4, ["&nbsp;\n", "## 练习 4.1：前馈模块与注意力模块中的参数"])
    
    # Cell 7
    set_source(7, [
        "- 上面的结果是针对单个 Transformer 块的\n",
        "- 可以选择乘以 12 以捕获 124M GPT 模型中的所有 Transformer 块"
    ])
    
    # Cell 8
    set_source(8, [
        "**奖励：数学分解**\n",
        "\n",
        "- 对于那些对如何通过数学计算这些参数数量感兴趣的人，可以在下面找到分解（假设 `emb_dim=768`）：\n",
        "\n",
        "\n",
        "前馈模块：\n",
        "\n",
        "- 第 1 个 `Linear` 层：768 输入 × 4×768 输出 + 4×768 偏置单元 = 2,362,368\n",
        "- 第 2 个 `Linear` 层：4×768 输入 × 768 输出 + 768 偏置单元 = 2,360,064\n",
        "- 总计：第 1 个 `Linear` 层 + 第 2 个 `Linear` 层 = 2,362,368 + 2,360,064 = 4,722,432\n",
        "\n",
        "注意力模块：\n",
        "\n",
        "- `W_query`：768 输入 × 768 输出 = 589,824 \n",
        "- `W_key`：768 输入 × 768 输出 = 589,824\n",
        "- `W_value`：768 输入 × 768 输出 = 589,824 \n",
        "- `out_proj`：768 输入 × 768 输出 + 768 偏置单元 = 590,592\n",
        "- 总计：`W_query` + `W_key` + `W_value` + `out_proj` = 3×589,824 + 590,592 = 2,360,064 "
    ])
    
    # Cell 9
    set_source(9, ["&nbsp;\n", "## 练习 4.2：初始化更大的 GPT 模型"])
    
    # Cell 10
    set_source(10, [
        "- **GPT2-small**（我们已经实现的 124M 配置）：\n",
        "    - \"emb_dim\" = 768\n",
        "    - \"n_layers\" = 12\n",
        "    - \"n_heads\" = 12\n",
        "\n",
        "- **GPT2-medium:**\n",
        "    - \"emb_dim\" = 1024\n",
        "    - \"n_layers\" = 24\n",
        "    - \"n_heads\" = 16\n",
        "\n",
        "- **GPT2-large:**\n",
        "    - \"emb_dim\" = 1280\n",
        "    - \"n_layers\" = 36\n",
        "    - \"n_heads\" = 20\n",
        "\n",
        "- **GPT2-XL:**\n",
        "    - \"emb_dim\" = 1600\n",
        "    - \"n_layers\" = 48\n",
        "    - \"n_heads\" = 25"
    ])
    
    # Cell 12
    set_source(12, ["&nbsp;\n", "## 练习 4.3：使用单独的 Dropout 参数"])

    print(f"Writing {target_path}...")
    with open(target_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Done.")

if __name__ == "__main__":
    translate_notebook()
