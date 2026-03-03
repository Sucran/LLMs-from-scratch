# 额外的分类微调实验

下表添加了实验来回答有关各种设计选择的其他问题。第一行使用与主要章节相同的设置，并用作参考。
例如，

- 比较第 1 行和第 2 行回答了这个问题：“当我们训练最后一个或第一个令牌时，性能差异是什么？”；
- 比较第 1 行和第 3 行回答了这个问题：“当我们只训练最后一层而不是最后一个块时，性能差异是什么？”；
- 等等。

&nbsp;

|      | 模型               | 权重       | 可训练令牌位置           | 可训练层         | 上下文长度                                             | 训练准确率   | 验证准确率     | 测试准确率 | 训练时间      | CPU/GPU |
| ---- | ------------------ | ---------- | ------------------------ | ---------------- | ------------------------------------------------------ | ------------ | -------------- | -------- | ------------- | ------- |
| 1    | gpt2-small (124M)  | 预训练     | 最后                     | last_block       | 最长训练示例 (120)                                     | 96.63%       | 99.33%         | 95.00%   | 0.28 min      | A100    |
| 2    | gpt2-small (124M)  | 预训练     | 第一                     | last_block       | 最长训练示例 (120)                                     | 78.46%       | 80.54%         | 75.00%   | 0.28 min      | A100    |
| 3    | gpt2-small (124M)  | 预训练     | 最后                     | last_layer       | 最长训练示例 (120)                                     | 78.65%       | 79.87%         | 72.00%   | 0.25 min      | A100    |
| 4    | gpt2-small (124M)  | 预训练     | 最后                     | last_two_blocks  | 最长训练示例 (120)                                     | 98.85%       | 98.66%         | 98.33%   | 0.33 min      | A100    |
| 5    | gpt2-small (124M)  | 预训练     | 最后                     | all              | 最长训练示例 (120)                                     | 99.62%       | 96.64%         | 96.67%   | 0.69 min      | A100    |
| 6    | gpt2-medium (355M) | 预训练     | 最后                     | last_block       | 最长训练示例 (120)                                     | 87.50%       | 91.28%         | 84.67%   | 0.75 min      | A100    |
| 7    | gpt2-large (774M)  | 预训练     | 最后                     | last_block       | 最长训练示例 (120)                                     | 99.52%       | 98.66%         | 96.67%   | 1.50 min      | A100    |
| 8    | gpt2-xl (1558M)    | 预训练     | 最后                     | last_block       | 最长训练示例 (120)                                     | 99.81%       | 99.81%         | 98.33%   | 2.83 min      | A100    |
| 9    | gpt2-xl (1558M)    | 预训练     | 最后                     | all              | 最长训练示例 (120)                                     | 100.00%      | 98.66%         | 98.67%   | 8.12 min      | A100    |
| 10   | gpt2-small (124M)  | 随机       | 最后                     | all              | 最长训练示例 (120)                                     | 100.00%      | 96.64%         | 93.67%   | 0.69 min      | A100    |
| 11   | gpt2-small (124M)  | 预训练     | 最后                     | LoRA             | 最长训练示例 (120)                                     | 100.00%      | 97.32%         | 96.67%   | 0.75 min      | A100    |
| 12   | gpt2-xl (1558M)    | 预训练     | 最后                     | LoRA             | 最长训练示例 (120)                                     | 100.00%      | 98.66%         | 98.33%   | 5.79 min      | A100    |
| 13   | gpt2-small (124M)  | 预训练     | 最后                     | last_block       | 上下文长度 (1024)                                      | 83.08%       | 87.92%         | 78.33%   | 2.46 min      | A100    |
| 14   | gpt2-small (124M)  | 预训练     | 最后                     | last_block       | 可变：无填充 (批量大小 1)                              | 100.00%      | 98.66%         | 98.00%   | 1.75 min      | A100    |
| 15   | gpt2-small (124M)  | 预训练     | 最后                     | last_block       | 可变：无填充 (批量大小 8)                              | 99.33%       | 98.66%         | 98.33%   | 1.70 min      | A100    |
| 16   | gpt2-small (124M)  | 预训练     | 最后                     | last_block       | 灵活 (最后一个非填充位置)                              | 99.42%       | 98.66%         | 98.33%   | 0.30 min      | A100    |
| 17   | gpt2-small (124M)  | 预训练     | 最后                     | last_block       | 最长训练示例 (120); 但无因果掩码                       | 99.23%       | 98.66%         | 95.33%   | 0.29 min      | A100    |
| 18   | gpt2-small (124M)  | 预训练     | 最后                     | last_block       | 最长训练示例 (120) 且 `ignore_index` 用于填充          | 96.63%       | 99.33%         | 95.00%   | 0.28 min      | A100    |
| 19   | gpt2-small (124M)  | 预训练     | 最后 + 池化嵌入          | last_block       | 最长训练示例 (120)                                     | 97.79%       | 99.33%         | 96.33%   | 0.32 min      | A100    |

&nbsp;

### 用法

你可以使用以下代码重现实验：

- 第 1 行：`python additional_experiments.py`
- 第 2 行：`python additional_experiments.py --trainable_token_pos first`
- 第 3 行：`python additional_experiments.py --trainable_layers last_layer`
- 第 4 行：`python additional_experiments.py --trainable_layers last_two_blocks`
- 第 5 行：`python additional_experiments.py --trainable_layers all`
- 第 6 行：`python additional_experiments.py --model_size "gpt2-medium (355M)"`
- 第 7 行：`python additional_experiments.py --model_size "gpt2-large (774M)"`
- 第 8 行：`python additional_experiments.py --model_size "gpt2-xl (1558M)"`
- 第 9 行：`python additional_experiments.py --model_size "gpt2-xl (1558M)"--trainable_layers all`
- 第 10 行：`python additional_experiments.py --weights random --trainable_layers all`
- 第 11 行：`python additional_experiments.py --trainable_layers lora --lora_rank 16 --lora_alpha 16`
- 第 12 行：`python additional_experiments.py --trainable_layers lora --lora_rank 16 --lora_alpha 8 --model_size "gpt2-xl (1558M)"`
- 第 13 行：`python additional_experiments.py --context_length "model_context_length"`
- 第 14 行：`python additional_experiments.py --no_padding --batch_size 1`
- 第 15 行：`python additional_experiments.py --no_padding --batch_size 1 --accumulation_steps 8`
- 第 16 行：`python additional_experiments.py --trainable_token_pos "flexible"`
- 第 17 行：`python additional_experiments.py --disable_causal_mask`
- 第 18 行：`python additional_experiments.py --ignore_index 50256`
- 第 19 行：`python additional_experiments.py --average_embeddings`

我故意保持 LLM 和数据集较小，以便你可以在大约 15 分钟内（对于默认设置）在像 MacBook Air M3 这样的常规笔记本电脑上运行训练，以防你无法使用 GPU。

&nbsp;

### 解释

1. **训练最后一个与第一个输出令牌位置（第 1 行与第 2 行）**：训练最后一个输出令牌位置会导致性能大大优于第一个。由于因果自注意力掩码，这种改进是预期的。
2. **训练最后一个 Transformer 块与最后一层（第 1 行与第 3 行）**：训练整个最后一个 transformer 块的结果也大大优于仅训练最后一层。
3. **训练最后两个 Transformer 块与最后一个（第 1 行与第 4 行）**：训练最后两个 transformer 块而不是仅训练最后一个块会导致明显的 3.33% 准确率提升。
4. **训练最后一个 Transformer 块与所有层（第 1 行与第 5 行）**：训练所有层显示出比仅训练最后一个 transformer 块有约 2% 的适度改进，但它在训练持续时间方面需要近三倍的时间。此外，它的表现不如仅训练 12 个 transformer 块中的最后两个。
5. **使用更大的预训练模型（第 1 行与第 6 行，以及第 1 行与第 7 和 8 行）**：使用 3 倍大的预训练模型会导致更差的结果。然而，使用 5 倍大的模型比初始模型提高了性能，正如预期的那样。同样，12 倍大的模型进一步提高了预测性能。（中型模型可能没有很好地预训练，或者特定的微调配置对于该模型效果不佳。）
6. **使用具有随机权重的模型与预训练权重（第 1 和 5 行与第 10 行）**：使用具有随机权重的模型产生的结果仅比使用预训练权重稍差（分别差 3% 和 1.3%）。
7. **使用 LoRA（低秩适应）与训练所有层（第 11 行与第 5 行，以及第 12 行与第 9 行）**：保持模型冻结并添加可训练的 LoRA 层（有关详细信息，请参阅 [附录 E](../../appendix-E/01_main-chapter-code/appendix-E.ipynb)）是训练所有模型参数的可行替代方案，甚至将性能提高了 1 个百分点（第 11 行与第 5 行）。从使用 LoRA 时训练和验证准确率之间约 1% 的较低差距可以看出，这可能是由于较少的过拟合。此外，使用 LoRA 也更节省内存，因为需要更新的参数更少。当训练较大的模型（第 12 行与第 9 行）时，我们还可以看到 LoRA 训练得更快（5.79 分钟而不是 8.12 分钟）。
8. **将输入填充到全上下文长度与最长训练示例（第 1 行与第 13 行）**：将输入填充到全支持的上下文长度结果明显更差。
9. **填充与无填充（第 1 行与第 14 和 15 行，以及第 16 行）**：`--no_padding` 选项禁用数据集中的填充，这需要使用批量大小为 1 训练模型，因为输入具有可变长度。这导致更好的测试准确率，但训练时间更长。在第 15 行中，我们另外启用了 8 步的梯度累积，以实现与其他实验相同的批量大小，这有助于减少过拟合并没有略微提高测试集准确率。在第 16 行中，应用了填充，但基于最后一个非填充令牌选择令牌位置。第 16 行在数学上应该类似于第 15 行，后者使用梯度累积。然而，由于在不等令牌计数的情况下梯度累积存在一些挑战，可能会有小的差异（这在 [这篇](https://unsloth.ai/blog/gradient) 博客文章中进行了讨论）。
10. **禁用因果注意力掩码（第 1 行与第 17 行）**：禁用多头注意力模块中使用的因果注意力掩码。这意味着所有令牌都可以关注所有其他令牌。与具有因果掩码的 GPT 模型相比，模型准确率略有提高。
11. **在损失和反向传播中忽略填充索引（第 1 行与第 18 行）**：设置 `--ignore_index 50256` 在 PyTorch 的 `cross_entropy` 损失函数中排除 `<|endoftext|>` 填充令牌。在这种情况下，它没有任何效果，因为我们替换了输出层，以便对于二元分类示例，令牌 ID 要么是 0 要么是 1。但是，此设置在第 7 章中的指令微调模型时很有用。
12. **平均所有令牌的嵌入（第 1 行与第 19 行）**：设置 `--average_embeddings` 将平均所有令牌的嵌入。如果不使用此选项（默认），则仅考虑所选令牌位置（由 `--trainable_token_pos` 指定）的输出嵌入；例如，最后一个令牌的嵌入。启用 `--average_embeddings` 将把所有令牌的嵌入平均池化到由 `--trainable_token_pos` 选择的位置（默认为最后一个令牌）。正如我们所看到的，这将性能从 95.00% 提高到 96.33%，而运行时间仅略有增加（从 0.28 分钟到 0.32 分钟），在实践中可能值得考虑。
