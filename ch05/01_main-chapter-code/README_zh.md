# 第 5 章：在未标记数据上进行预训练

### 主要章节代码

- [ch05.ipynb](ch05.ipynb) 包含本章中出现的所有代码
- [previous_chapters.py](previous_chapters.py) 是一个 Python 模块，包含上一章的 `MultiHeadAttention` 模块和 `GPTModel` 类，我们在 [ch05.ipynb](ch05.ipynb) 中导入它以预训练 GPT 模型
- [gpt_download.py](gpt_download.py) 包含用于下载预训练 GPT 模型权重的实用程序函数
- [exercise-solutions.ipynb](exercise-solutions.ipynb) 包含本章的练习解答

### 可选代码

- [gpt_train.py](gpt_train.py) 是一个独立的 Python 脚本文件，包含我们在 [ch05.ipynb](ch05.ipynb) 中实现的用于训练 GPT 模型的代码（你可以将其视为总结本章的代码文件）
- [gpt_generate.py](gpt_generate.py) 是一个独立的 Python 脚本文件，包含我们在 [ch05.ipynb](ch05.ipynb) 中实现的用于从 OpenAI 加载和使用预训练模型权重的代码
