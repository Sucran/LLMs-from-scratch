# 从头构建大型语言模型 (Build a Large Language Model From Scratch)

本仓库包含了开发、预训练和微调类似 GPT 的大型语言模型（LLM）的代码，是书籍 [《从头构建大型语言模型》](https://amzn.to/4fqvn0D) 的官方代码仓库。

<br>
<br>

<a href="https://amzn.to/4fqvn0D"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/cover.jpg?123" width="250px"></a>

<br>

在 [《从头构建大型语言模型》](http://mng.bz/orYv) 中，你将通过从零开始一步步编写代码，深入了解和掌握大型语言模型（LLMs）的内部工作原理。在本书中，我将指导你创建自己的 LLM，并通过清晰的文字、图表和示例解释每个阶段。

本书中描述的用于训练和开发用于教育目的的小型但功能齐全的模型的方法，反映了创建诸如 ChatGPT 背后的那些大规模基础模型所使用的方法。此外，本书还包含了加载用于微调的较大型预训练模型权重的代码。

- 官方 [源代码仓库](https://github.com/rasbt/LLMs-from-scratch) 链接
- [Manning（出版商网站）上的书籍链接](http://mng.bz/orYv)
- [Amazon.com 上的书籍页面链接](https://www.amazon.com/gp/product/1633437167)
- ISBN 9781633437166

<a href="http://mng.bz/orYv#reviews"><img src="https://sebastianraschka.com//images/LLMs-from-scratch-images/other/reviews.png" width="220px"></a>


<br>
<br>

要下载本仓库的副本，请点击 [Download ZIP](https://github.com/rasbt/LLMs-from-scratch/archive/refs/heads/main.zip) 按钮，或在终端中执行以下命令：

```bash
git clone --depth 1 https://github.com/rasbt/LLMs-from-scratch.git
```

<br>

（如果你是从 Manning 网站下载的代码包，请考虑访问 GitHub 上的官方代码仓库 [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) 以获取最新更新。）

<br>
<br>


# 目录

请注意，此 `README.md` 文件是一个 Markdown (`.md`) 文件。如果你是从 Manning 网站下载的代码包并在本地计算机上查看，我建议使用 Markdown 编辑器或预览器以获得正确的查看体验。如果你还没有安装 Markdown 编辑器，[Ghostwriter](https://ghostwriter.kde.org) 是一个不错的免费选择。

或者，你也可以在浏览器中访问 GitHub 上的 [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) 查看此文件和其他文件，它会自动渲染 Markdown。

<br>
<br>


> **提示：**
> 如果你正在寻找关于安装 Python 和 Python 包以及设置代码环境的指导，我建议阅读位于 [setup](setup) 目录下的 [README.md](setup/README.md) 文件。

<br>
<br>

[![Code tests Linux](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-linux-uv.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-linux-uv.yml)
[![Code tests Windows](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-windows-uv-pip.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-windows-uv-pip.yml)
[![Code tests macOS](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-macos-uv.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-macos-uv.yml)



| 章节标题                                                   | 主要代码（用于快速访问）                                                                                                          | 所有代码 + 补充材料           |
|------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| [设置建议](setup) <br/>[如何最好地阅读本书](https://sebastianraschka.com/blog/2025/reading-books.html)                            | -                                                                                                                               | -                             |
| 第 1 章：理解大型语言模型                                  | 无代码                                                                                                                          | -                             |
| 第 2 章：处理文本数据                                      | - [ch02.ipynb](ch02/01_main-chapter-code/ch02.ipynb)<br/>- [dataloader.ipynb](ch02/01_main-chapter-code/dataloader.ipynb) (摘要)<br/>- [exercise-solutions.ipynb](ch02/01_main-chapter-code/exercise-solutions.ipynb)               | [./ch02](./ch02)            |
| 第 3 章：编写注意力机制代码                                | - [ch03.ipynb](ch03/01_main-chapter-code/ch03.ipynb)<br/>- [multihead-attention.ipynb](ch03/01_main-chapter-code/multihead-attention.ipynb) (摘要) <br/>- [exercise-solutions.ipynb](ch03/01_main-chapter-code/exercise-solutions.ipynb)| [./ch03](./ch03)             |
| 第 4 章：从头实现 GPT 模型                                 | - [ch04.ipynb](ch04/01_main-chapter-code/ch04.ipynb)<br/>- [gpt.py](ch04/01_main-chapter-code/gpt.py) (摘要)<br/>- [exercise-solutions.ipynb](ch04/01_main-chapter-code/exercise-solutions.ipynb) | [./ch04](./ch04)           |
| 第 5 章：在未标记数据上进行预训练                          | - [ch05.ipynb](ch05/01_main-chapter-code/ch05.ipynb)<br/>- [gpt_train.py](ch05/01_main-chapter-code/gpt_train.py) (摘要) <br/>- [gpt_generate.py](ch05/01_main-chapter-code/gpt_generate.py) (摘要) <br/>- [exercise-solutions.ipynb](ch05/01_main-chapter-code/exercise-solutions.ipynb) | [./ch05](./ch05)              |
| 第 6 章：微调用于文本分类                                  | - [ch06.ipynb](ch06/01_main-chapter-code/ch06.ipynb)  <br/>- [gpt_class_finetune.py](ch06/01_main-chapter-code/gpt_class_finetune.py)  <br/>- [exercise-solutions.ipynb](ch06/01_main-chapter-code/exercise-solutions.ipynb) | [./ch06](./ch06)              |
| 第 7 章：微调以遵循指令                                    | - [ch07.ipynb](ch07/01_main-chapter-code/ch07.ipynb)<br/>- [gpt_instruction_finetuning.py](ch07/01_main-chapter-code/gpt_instruction_finetuning.py) (摘要)<br/>- [ollama_evaluate.py](ch07/01_main-chapter-code/ollama_evaluate.py) (摘要)<br/>- [exercise-solutions.ipynb](ch07/01_main-chapter-code/exercise-solutions.ipynb) | [./ch07](./ch07)  |
| 附录 A：PyTorch 简介                                       | - [code-part1.ipynb](appendix-A/01_main-chapter-code/code-part1.ipynb)<br/>- [code-part2.ipynb](appendix-A/01_main-chapter-code/code-part2.ipynb)<br/>- [DDP-script.py](appendix-A/01_main-chapter-code/DDP-script.py)<br/>- [exercise-solutions.ipynb](appendix-A/01_main-chapter-code/exercise-solutions.ipynb) | [./appendix-A](./appendix-A) |
| 附录 B：参考文献和进一步阅读                               | 无代码                                                                                                                          | [./appendix-B](./appendix-B) |
| 附录 C：练习解答                                           | - [练习解答列表](appendix-C)                                                                 | [./appendix-C](./appendix-C) |
| 附录 D：为训练循环添加额外功能                             | - [appendix-D.ipynb](appendix-D/01_main-chapter-code/appendix-D.ipynb)                                                          | [./appendix-D](./appendix-D)  |
| 附录 E：使用 LoRA 进行参数高效微调                         | - [appendix-E.ipynb](appendix-E/01_main-chapter-code/appendix-E.ipynb)                                                          | [./appendix-E](./appendix-E) |

<br>
&nbsp;

下面的思维模型总结了本书涵盖的内容。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/mental-model.jpg" width="650px">


<br>
&nbsp;

## 先决条件

最重要的先决条件是扎实的 Python 编程基础。
有了这些知识，你将做好准备去探索 LLM 的迷人世界，并理解本书中介绍的概念和代码示例。

如果你有一些深度神经网络的经验，你会发现某些概念更加熟悉，因为 LLM 是建立在这些架构之上的。

本书使用 PyTorch 从头开始实现代码，不使用任何外部 LLM 库。虽然精通 PyTorch 不是先决条件，但熟悉 PyTorch 基础知识肯定是有用的。如果你是 PyTorch 新手，附录 A 提供了一个简明的 PyTorch 介绍。或者，你可能会发现我的书 [《PyTorch in One Hour: From Tensors to Training Neural Networks on Multiple GPUs》](https://sebastianraschka.com/teaching/pytorch-1h/) 对学习基础知识很有帮助。



<br>
&nbsp;

## 硬件要求

本书主要章节中的代码旨在在传统笔记本电脑上在合理的时间内运行，不需要专门的硬件。这种方法确保了广大读者都能参与到材料中。此外，如果有可用的 GPU，代码会自动利用它们。（请参阅 [setup](https://github.com/rasbt/LLMs-from-scratch/blob/main/setup/README.md) 文档以获取更多建议。）


&nbsp;
## 视频课程

[一个 17 小时 15 分钟的配套视频课程](https://www.manning.com/livevideo/master-and-build-large-language-models)，我在其中编写了本书每一章的代码。课程分为章节和小节，反映了本书的结构，因此它可以作为本书的独立替代品或补充的跟随编码资源。

<a href="https://www.manning.com/livevideo/master-and-build-large-language-models"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/video-screenshot.webp?123" width="350px"></a>


&nbsp;


## 配套书籍 / 续集

[*Build A Reasoning Model (From Scratch)*](https://mng.bz/lZ5B)，虽然是一本独立的书，但可以被视为 *Build A Large Language Model (From Scratch)* 的续集。

它从一个预训练模型开始，实现了不同的推理方法，包括推理时扩展、强化学习和蒸馏，以提高模型的推理能力。

与 *Build A Large Language Model (From Scratch)* 类似，[*Build A Reasoning Model (From Scratch)*](https://mng.bz/lZ5B) 采用动手实践的方法，从头开始实现这些方法。

<a href="https://mng.bz/lZ5B"><img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/cover.webp?123" width="120px"></a>

- Amazon 链接 (待定)
- [Manning 链接](https://mng.bz/lZ5B)
- [GitHub 仓库](https://github.com/rasbt/reasoning-from-scratch)

<br>

&nbsp;
## 练习

本书的每一章都包含几个练习。解答在附录 C 中进行了总结，相应的代码笔记本可在本仓库的主要章节文件夹中找到（例如，[./ch02/01_main-chapter-code/exercise-solutions.ipynb](./ch02/01_main-chapter-code/exercise-solutions.ipynb)）。

除了代码练习外，你还可以从 Manning 网站下载名为 [Test Yourself On Build a Large Language Model (From Scratch)](https://www.manning.com/books/test-yourself-on-build-a-large-language-model-from-scratch) 的免费 170 页 PDF。它包含每章大约 30 个测验问题和解答，以帮助你测试你的理解。

<a href="https://www.manning.com/books/test-yourself-on-build-a-large-language-model-from-scratch"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/test-yourself-cover.jpg?123" width="150px"></a>

&nbsp;
## 奖励材料

几个文件夹包含供感兴趣的读者使用的可选奖励材料：
- **设置**
  - [Python 设置技巧](setup/01_optional-python-setup-preferences)
  - [安装本书中使用的 Python 包和库](setup/02_installing-python-libraries)
  - [Docker 环境设置指南](setup/03_optional-docker-environment)

- **第 2 章：处理文本数据**
  - [从头开始的字节对编码 (BPE) 分词器](ch02/05_bpe-from-scratch/bpe-from-scratch-simple.ipynb)
  - [比较各种字节对编码 (BPE) 实现](ch02/02_bonus_bytepair-encoder)
  - [理解嵌入层和线性层之间的区别](ch02/03_bonus_embedding-vs-matmul)
  - [使用简单数字的数据加载器直觉](ch02/04_bonus_dataloader-intuition)

- **第 3 章：编写注意力机制代码**
  - [比较高效的多头注意力实现](ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb)
  - [理解 PyTorch 缓冲区](ch03/03_understanding-buffers/understanding-buffers.ipynb)

- **第 4 章：从头实现 GPT 模型**
  - [FLOPs 分析](ch04/02_performance-analysis/flops-analysis.ipynb)
  - [KV 缓存](ch04/03_kv-cache)
  - [注意力替代方案](ch04/#attention-alternatives)
    - [分组查询注意力 (GQA)](ch04/04_gqa)
    - [多头潜在注意力 (MLA)](ch04/05_mla)
    - [滑动窗口注意力 (SWA)](ch04/06_swa)
    - [门控 DeltaNet](ch04/08_deltanet)
  - [混合专家 (MoE)](ch04/07_moe)

- **第 5 章：在未标记数据上进行预训练**
  - [替代权重加载方法](ch05/02_alternative_weight_loading/)
  - [在 Project Gutenberg 数据集上预训练 GPT](ch05/03_bonus_pretraining_on_gutenberg)
  - [为训练循环添加额外功能](ch05/04_learning_rate_schedulers)
  - [优化预训练的超参数](ch05/05_bonus_hparam_tuning)
  - [构建与预训练 LLM 交互的用户界面](ch05/06_user_interface)
  - [将 GPT 转换为 Llama](ch05/07_gpt_to_llama)
  - [内存高效的模型权重加载](ch05/08_memory_efficient_weight_loading/memory-efficient-state-dict.ipynb)
  - [使用新令牌扩展 Tiktoken BPE 分词器](ch05/09_extending-tokenizers/extend-tiktoken.ipynb)
  - [用于更快 LLM 训练的 PyTorch 性能技巧](ch05/10_llm-training-speed)
  - [LLM 架构](ch05/#llm-architectures-from-scratch)
    - [从头开始的 Llama 3.2](ch05/07_gpt_to_llama/standalone-llama32.ipynb)
    - [从头开始的 Qwen3 密集和混合专家 (MoE)](ch05/11_qwen3/)
    - [从头开始的 Gemma 3](ch05/12_gemma3/)
    - [从头开始的 Olmo 3](ch05/13_olmo3/)
    - [从头开始的 Tiny Aya](ch05/15_tiny-aya/)
  - [第 5 章与其他 LLM 作为直接替换（例如，Llama 3, Qwen 3）](ch05/14_ch05_with_other_llms/)
- **第 6 章：微调用于分类**
  - [微调不同层和使用更大模型的额外实验](ch06/02_bonus_additional-experiments)
  - [在 50k IMDb 电影评论数据集上微调不同模型](ch06/03_bonus_imdb-classification)
  - [构建与基于 GPT 的垃圾邮件分类器交互的用户界面](ch06/04_user_interface)
- **第 7 章：微调以遵循指令**
  - [用于查找近似重复项和创建被动语态条目的数据集实用程序](ch07/02_dataset-utilities)
  - [使用 OpenAI API 和 Ollama 评估指令响应](ch07/03_model-evaluation)
  - [生成用于指令微调的数据集](ch07/05_dataset-generation/llama3-ollama.ipynb)
  - [改进用于指令微调的数据集](ch07/05_dataset-generation/reflection-gpt4.ipynb)
  - [使用 Llama 3.1 70B 和 Ollama 生成偏好数据集](ch07/04_preference-tuning-with-dpo/create-preference-data-ollama.ipynb)
  - [用于 LLM 对齐的直接偏好优化 (DPO)](ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb)
  - [构建与指令微调后的 GPT 模型交互的用户界面](ch07/06_user_interface)

来自 [Reasoning From Scratch](https://github.com/rasbt/reasoning-from-scratch) 仓库的更多奖励材料：

- **Qwen3 (从头开始) 基础**
  - [Qwen3 源代码演练](https://github.com/rasbt/reasoning-from-scratch/blob/main/chC/01_main-chapter-code/chC_main.ipynb)
  - [优化的 Qwen3](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch02/03_optimized-LLM)

- **评估**
  - [基于验证者的评估 (MATH-500)](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch03)
  - [多项选择评估 (MMLU)](https://github.com/rasbt/reasoning-from-scratch/blob/main/chF/02_mmlu)
  - [LLM 排行榜评估](https://github.com/rasbt/reasoning-from-scratch/blob/main/chF/03_leaderboards)
  - [LLM 作为裁判评估](https://github.com/rasbt/reasoning-from-scratch/blob/main/chF/04_llm-judge)
- **推理扩展**
  - [自洽性](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch04/01_main-chapter-code/ch04_main.ipynb)
  - [自我完善](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch05/01_main-chapter-code/ch05_main.ipynb)

- **强化学习** (RL)
  - [从头开始使用 GRPO 的 RLVR](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch06/01_main-chapter-code/ch06_main.ipynb)


<br>
&nbsp;

## 问题、反馈和贡献

我欢迎各种反馈，最好通过 [Manning 论坛](https://livebook.manning.com/forum?product=raschka&page=1) 或 [GitHub Discussions](https://github.com/rasbt/LLMs-from-scratch/discussions) 分享。同样，如果你有任何问题或只是想与他人交流想法，请随时在论坛上发帖。

请注意，由于本仓库包含对应于印刷书籍的代码，目前我无法接受扩展主要章节代码内容的贡献，因为这会引入与实体书的偏差。保持一致有助于确保每个人的流畅体验。


&nbsp;
## 引用

如果你发现本书或代码对你的研究有用，请考虑引用它。

芝加哥格式引用：

> Raschka, Sebastian. *Build A Large Language Model (From Scratch)*. Manning, 2024. ISBN: 978-1633437166.

BibTeX 条目：

```
@book{build-llms-from-scratch-book,
  author       = {Sebastian Raschka},
  title        = {Build A Large Language Model (From Scratch)},
  publisher    = {Manning},
  year         = {2024},
  isbn         = {978-1633437166},
  url          = {https://www.manning.com/books/build-a-large-language-model-from-scratch},
  github       = {https://github.com/rasbt/LLMs-from-scratch}
}
```
