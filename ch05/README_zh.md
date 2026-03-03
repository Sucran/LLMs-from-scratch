# 第 5 章：在未标记数据上进行预训练

&nbsp;
## 主要章节代码

- [01_main-chapter-code](01_main-chapter-code) 包含主要章节代码

&nbsp;
## 奖励材料

- [02_alternative_weight_loading](02_alternative_weight_loading) 包含从替代位置加载 GPT 模型权重的代码，以防无法从 OpenAI 获取模型权重
- [03_bonus_pretraining_on_gutenberg](03_bonus_pretraining_on_gutenberg) 包含在古腾堡计划 (Project Gutenberg) 的整本书籍语料库上对 LLM 进行更长时间预训练的代码
- [04_learning_rate_schedulers](04_learning_rate_schedulers) 包含实现更复杂的训练函数的代码，包括学习率调度器和梯度裁剪
- [05_bonus_hparam_tuning](05_bonus_hparam_tuning) 包含一个可选的超参数调整脚本
- [06_user_interface](06_user_interface) 实现了一个交互式用户界面，以便与预训练的 LLM 进行交互
- [08_memory_efficient_weight_loading](08_memory_efficient_weight_loading) 包含一个奖励笔记本，展示如何通过 PyTorch 的 `load_state_dict` 方法更高效地加载模型权重
- [09_extending-tokenizers](09_extending-tokenizers) 包含从头开始实现的 GPT-2 BPE 分词器
- [10_llm-training-speed](10_llm-training-speed) 展示了提高 LLM 训练速度的 PyTorch 性能技巧

&nbsp;
## 从头开始的 LLM 架构

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen-overview.webp">

&nbsp;


- [07_gpt_to_llama](07_gpt_to_llama) 包含将 GPT 架构实现转换为 Llama 3.2 并从 Meta AI 加载预训练权重的分步指南
- [11_qwen3](11_qwen3) 从头开始实现 Qwen3 0.6B 和 Qwen3 30B-A3B（混合专家），包括加载基础、推理和编码模型变体的预训练权重的代码
- [12_gemma3](12_gemma3) 从头开始实现 Gemma 3 270M 和带 KV 缓存的替代方案，包括加载预训练权重的代码
- [13_olmo3](13_olmo3) 从头开始实现 Olmo 3 7B 和 32B（基础、指令和思考变体）和带 KV 缓存的替代方案，包括加载预训练权重的代码

&nbsp;
## 本章的代码演示视频

<br>
<br>

[![Link to the video](https://img.youtube.com/vi/Zar2TJv-sE0/0.jpg)](https://www.youtube.com/watch?v=Zar2TJv-sE0)
