# 混合专家 (MoE)

本奖励材料说明了使用混合专家 (MoE) 层代替常规前馈 (FFN) 层时的内存节省（每个令牌）。



&nbsp;
## 介绍

MoE 的核心思想是将 transformer 块中的每个前馈模块替换为多个专家层，其中每个专家层也是一个前馈模块。这意味着我们将单个前馈块替换为多个前馈块，如下图所示。



&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/1.webp" alt="SWA" width="800px" />

transformer 块内的前馈块（在上图中显示为深灰色块）通常包含模型总参数的大量部分。（请注意，transformer 块，从而前馈块，在 LLM 中重复多次；在 DeepSeek-V3 的情况下，为 61 次。）

因此，用 *多个* 前馈块替换 *单个* 前馈块（如在 MoE 设置中所做的那样）会大大增加模型的总参数数量。然而，关键的技巧是我们不对每个令牌使用（“激活”）所有专家。相反，路由器每个令牌只选择一小部分专家。

因为一次只有少数专家处于活动状态，所以 MoE 模块通常被称为 *稀疏*，与始终使用完整参数集的 *密集* 模块形成对比。然而，通过 MoE 的大量参数增加了 LLM 的容量，这意味着它可以在训练期间吸收更多知识。稀疏性保持了推理的高效，因为我们不同时使用所有参数。

例如，DeepSeek-V3 每个 MoE 模块有 256 个专家，总共有 6710 亿个参数。然而在推理期间，一次只有 9 个专家处于活动状态（1 个共享专家加上 8 个由路由器选择的专家）。这意味着每个令牌推理步骤只使用了 370 亿个参数，而不是全部 6710 亿个。

DeepSeek-V3 的 MoE 设计的一个显着特点是使用了共享专家。这是一个对每个令牌都始终处于活动状态的专家。这个想法并不新鲜，已经在 [2022 DeepSpeed-MoE](https://arxiv.org/abs/2201.05596) 和 [2024 DeepSeek MoE](https://arxiv.org/abs/2401.06066) 论文中介绍过。

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/3.webp?1" alt="MoE shared expert" width="500px" />

（来自 [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) 论文的注释图。）

&nbsp;

拥有共享专家的好处首先在 [DeepSpeed-MoE 论文](https://arxiv.org/abs/2201.05596) 中被注意到，他们发现与没有共享专家相比，它提高了整体建模性能。这可能是因为常见或重复的模式不必由多个单独的专家学习，这为他们留出了更多空间来学习更专业的模式。

&nbsp;
## 混合专家 (MoE) 内存节省

MoE 模型中的内存节省主要来自减少的激活存储和计算。在常规（密集）前馈层 (FFN) 中，每个令牌都会激活完整的中间维度。

相比之下，MoE 层每个令牌仅通过一小部分专家（例如，`num_experts` 中的 `top_k`）路由每个令牌。

当使用 MoE 层时，每个令牌只有 `top_k` 个专家处于活动状态，因此相对于具有相同总容量的密集 FFN，有效内存（和计算）大约缩放了 `top_k / num_experts` 的因子。


你可以使用此文件夹中的 [memory_estimator_moe.py](memory_estimator_moe.py) 脚本将其应用于不同的模型配置，以查看通过使用 MoE 而不是 FFN 可以节省多少内存（请注意，这是针对单个 transformer 块的，要获得总节省，请乘以模型中的 transformer 块数）：

```bash
uv run memory_estimator_moe.py --emb_dim 7168 --hidden_dim 14336 --ffn_type swiglu \
  --num_experts 8 --top_k 2 --match_dense 
==== Config ====
emb_dim                : 7168
hidden_size            : 14336
ffn_type               : swiglu
num_experts            : 8
top_k                  : 2
dtype                  : bf16 (2 Bytes/elem)
match_dense            : True

==== Model weights (parameters) ====
Dense FFN params       : 308,281,344 (0.62 GB)
Per-expert params      : 38,535,168 (0.08 GB)
Router params          : 57,344 (0.00 GB)
MoE TOTAL params       : 308,338,688 (0.62 GB)
MoE ACTIVE/Token       : 77,127,680 (0.15 GB)
moe_hidden_size        : 1792
```

所以，根据上面的结果，我们可以看到，如果我们有一个输入/输出维度 (`emb_dim`) 为 7,168 和中间大小 (`hidden_dim`) 为 14,336 的 FFN，我们在这一层中有 ~308M 参数，并且所有这些参数在前向传播中都是活动的。

现在，如果我们使用一个具有大致相同总参数数量 (~308M) 的 MoE 层，有 8 个专家，其中 2 个专家处于活动状态，那么在每次前向传播中只有 ~77M 参数处于活动状态。

此外，在专家数量不变的情况下，我们拥有的专家越多，活动参数的数量就越少，“节省”就越大：

&nbsp;

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/2.webp" alt="SWA" width="500px" />



&nbsp;

你可以通过以下方式重现此图：

```bash
uv run plot_memory_estimates_moe.py \
    --emb_dim 7168 \
    --hidden_dim 28672 \
    --ffn_type swiglu \
    --top_k 8
```


&nbsp;
## MoE 代码示例

此文件夹中的 [gpt_with_kv_ffn.py](gpt_with_kv_ffn.py) 和 [gpt_with_kv_moe.py](gpt_with_kv_moe.py) 脚本提供了在 GPT 模型实现的背景下比较常规 FFN 和 MoE 内存使用的动手示例。请注意，这两个脚本都使用 [SwiGLU](https://arxiv.org/abs/2002.05202) 前馈模块，如本页第一张图所示（GPT-2 传统上使用 GELU）。

**注意：该模型未经过训练，因此会生成无意义的文本。你可以在 [../../ch05/11_qwen3/standalone-qwen3-moe-plus-kvcache.ipynb](../../ch05/11_qwen3/standalone-qwen3-moe-plus-kvcache.ipynb) 的奖励材料中找到经过训练的 MoE。**



首先，让我们运行带有常规 FFN 的模型：


```bash
uv run gpt_with_kv_ffn.py \
--max_new_tokens 1024 \
--n_heads 16 \
--n_layers 12 \
--emb_dim 4096 \
--hidden_dim 32768

...
Avg FFN time/call: 0.759 ms
Avg FFN mem delta/call: 0.19 MB (max 0.75 MB)
...
Time: 25.13 sec
40 tokens/sec
Max memory allocated: 11.47 GB
```

为了与 MoE 进行公平比较，我们必须缩小专家规模。例如，如果我们使用 32 个专家，我们必须设置 `--hidden_dim 32768/32`：


```bash
uv run gpt_with_kv_moe.py \
--max_new_tokens 1024 \
--n_heads 16 \
--n_layers 12 \
--emb_dim 4096 \
--hidden_dim 1024 \
--num_experts 32 \
--num_experts_per_tok 2

...
Avg MoE FF time/call: 1.555 ms
Avg MoE FF mem delta/call: 0.04 MB (max 0.11 MB)
...
Time: 35.11 sec
29 tokens/sec
Max memory allocated: 11.48 GB
```

我们可以看到，密集前馈层处理一个令牌大约需要 0.76 毫秒，并使用大约 0.19 MB 的激活（峰值接近 0.75 MB）。

稀疏 MoE 层仅保留约 0.04 MB 的内存（峰值为 0.11）。然而，这是以大约两倍的计算时间为代价的。（增加了路由开销，我的实现可能也不是最有效的。）

在这两种情况下，整体生成仍然在 11.5 GB 的 GPU 内存左右达到峰值，因为两个版本加载相同数量的权重参数并具有相同的 KV 缓存大小，这在这里占主导地位。

无论哪种方式，我们都可以在这里看到权衡，MoE 将 FFN 内存减少了约 4-5 倍，同时大致加倍了前馈计算时间。

请注意，如果我们一次处理更多令牌，例如，批量大小大于 1（这里由于代码简单性我们没有批次），节省将更加明显。
