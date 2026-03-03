# 奖励材料：KV 缓存



**此文件夹实现了向 GPT 模型添加 KV 缓存。**

&nbsp;
## 概述

简而言之，KV 缓存存储中间键 (K) 和值 (V) 计算，以便在推理期间重用，这在生成响应时会导致显著的速度提升。缺点是它增加了一些代码复杂性，增加了内存使用量，并且不能在训练期间使用。然而，在部署 LLM 时，推理速度的提升通常非常值得在代码复杂性和内存方面进行权衡。

&nbsp;
## 工作原理

想象一下 LLM 正在生成一些文本。具体来说，假设 LLM 被赋予以下提示："Time flies"。

下图显示了使用第 3 章修改后的图形计算底层注意力分数的摘录，其中突出显示了键和值向量：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/kv-cache-attn-1.png?3" width=800>

现在，正如我们在第 2 章和第 4 章中学到的，LLM 一次生成一个单词（或令牌）。假设 LLM 生成了单词 "fast"，因此下一轮的提示变为 "Time flies fast"。这在下图中说明：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/kv-cache-attn-2.png?3" width=800>

正如我们所见，通过比较前两张图，前两个令牌的键和值向量完全相同，在每个下一个令牌文本生成轮次中重新计算它们将是一种浪费。

因此，KV 缓存的想法是实现一种缓存机制，存储先前生成的键和值向量以供重用，这有助于我们避免不必要的重新计算。

&nbsp;

## KV 缓存实现

有很多方法可以实现 KV 缓存，主要思想是我们只计算每个生成步骤中新生成令牌的键和值张量。

我选择了一个强调代码可读性的简单方法。我认为最简单的方法就是滚动浏览代码更改以查看它是如何实现的。

此文件夹中有两个文件：

1. [`gpt_ch04.py`](gpt_ch04.py)：取自第 3 章和第 4 章的自包含代码，用于实现 LLM 并运行简单的文本生成函数
2. [`gpt_with_kv_cache.py`](gpt_with_kv_cache.py)：与上面相同，但进行了必要的更改以实现 KV 缓存。

你可以

a. 打开 [`gpt_with_kv_cache.py`](gpt_with_kv_cache.py) 文件并查找标记新更改的 `# NEW` 部分：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/new-sections.png?3" width=800>

b. 通过你选择的文件差异工具查看这两个代码文件以比较更改：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/file-diff.png?3" width=800>

为了总结实现细节，这里有一个简短的演练。

&nbsp;

### 1. 注册缓存缓冲区

在 `MultiHeadAttention` 构造函数中，我们添加了两个缓冲区 `cache_k` 和 `cache_v`，它们将保存跨步骤的串联键和值：

```python
self.register_buffer("cache_k", None)
self.register_buffer("cache_v", None)
```

&nbsp;

### 2. 带 `use_cache` 标志的前向传播

接下来，我们扩展 `MultiHeadAttention` 类的 `forward` 方法以接受 `use_cache` 参数。在将新令牌块投影到 `keys_new`、`values_new` 和 `queries` 之后，我们要么初始化 kv 缓存，要么追加到我们的缓存：

```python
def forward(self, x, use_cache=False):
    b, num_tokens, d_in = x.shape

    keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
    values_new = self.W_value(x)
    queries = self.W_query(x)
    #...

    if use_cache:
        if self.cache_k is None:
            self.cache_k, self.cache_v = keys_new, values_new
        else:
            self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
            self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
        keys, values = self.cache_k, self.cache_v
    else:
        keys, values = keys_new, values_new
        
    # ...
    
    num_tokens_Q = queries.shape[-2]
    num_tokens_K = keys.shape[-2]
    if use_cache:
        mask_bool = self.mask.bool()[
            self.ptr_current_pos:self.ptr_current_pos + num_tokens_Q, :num_tokens_K
        ]
        self.ptr_current_pos += num_tokens_Q
    else:
        mask_bool = self.mask.bool()[:num_tokens_Q, :num_tokens_K]
```

&nbsp;


### 3. 清除缓存

生成文本时，在独立序列之间（例如文本生成调用），我们必须重置两个缓冲区，因此我们还将缓存重置方法添加到 `MultiHeadAttention` 类中：

```python
def reset_cache(self):
    self.cache_k, self.cache_v = None, None
    self.ptr_current_pos = 0
```

&nbsp;

### 4. 在完整模型中传播 `use_cache`

随着 `MultiHeadAttention` 类更改的到位，我们现在修改 `GPTModel` 类。首先，我们将令牌索引的位置跟踪添加到构造函数中：

```python
self.current_pos = 0
```

然后，我们将单行块调用替换为显式循环，通过每个 transformer 块传递 `use_cache`：

```python
def forward(self, in_idx, use_cache=False):
    # ...
 
    if use_cache:
        pos_ids = torch.arange(
            self.current_pos, self.current_pos + seq_len,            
            device=in_idx.device, dtype=torch.long
        )
        self.current_pos += seq_len
    else:
        pos_ids = torch.arange(
            0, seq_len, device=in_idx.device, dtype=torch.long
        )
    
    pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
    x = tok_embeds + pos_embeds
    # ...
    for blk in self.trf_blocks:
        x = blk(x, use_cache=use_cache)
```

上述更改还需要对 `TransformerBlock` 类进行少量修改以接受 `use_cache` 参数：
```python
    def forward(self, x, use_cache=False):
        # ...
        self.att(x, use_cache=use_cache)
```

最后，我们向 `GPTModel` 添加一个模型级重置 `reset_kv_cache`，以便为方便起见一次清除所有块缓存：

```python
def reset_kv_cache(self):
    for blk in self.trf_blocks:
        blk.att.reset_cache()
    self.current_pos = 0
```

&nbsp;

### 5. 在生成中使用缓存

随着对 `GPTModel`、`TransformerBlock` 和 `MultiHeadAttention` 的更改，最后，这是我们在简单文本生成函数中使用 KV 缓存的方式：

```python
def generate_text_simple_cached(model, idx, max_new_tokens, 
                                context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        if use_cache:
            # Init cache with full prompt
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability (greedy sampling)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) append it to the running sequence
                idx = torch.cat([idx, next_idx], dim=1)
                # c) feed model only the new token
                logits = model(next_idx, use_cache=True)
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx
```

请注意，我们仅通过 `logits = model(next_idx, use_cache=True)` 将新令牌输入模型 c)。如果不使用缓存，我们将整个输入 `logits = model(idx[:, -ctx_len:], use_cache=False)` 输入模型，因为它没有存储的键和值可重用。

&nbsp;

## 简单性能比较

在概念层面上涵盖 KV 缓存之后，最大的问题是它在一个小例子中的实际表现如何。为了尝试实现，我们可以将上述两个代码文件作为 Python 脚本运行，这将运行小型 124 M 参数 LLM 生成 200 个新令牌（给定一个 4 令牌提示 "Hello, I am" 开始）：

```bash
pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt

python gpt_ch04.py

python gpt_with_kv_cache.py
```

在带有 M4 芯片 (CPU) 的 Mac Mini 上，结果如下：

|                        | Tokens/sec |
| ---------------------- | ---------- |
| `gpt_ch04.py`          | 27         |
| `gpt_with_kv_cache.py` | 144        |

所以，正如我们所看到的，对于一个 124 M 参数的小型模型和一个 200 个令牌的短序列长度，我们已经获得了约 5 倍的加速。（请注意，此实现针对代码可读性进行了优化，并未针对 CUDA 或 MPS 运行时速度进行优化，这需要预分配张量而不是恢复和连接它们。）

**注意：** 模型在两种情况下都会生成“乱码”，即如下所示的文本：

> Output text: Hello, I am Featureiman Byeswickattribute argue logger Normandy Compton analogous bore ITVEGIN ministriesysics Kle functional recountrictionchangingVirgin embarrassedgl ...

这是因为我们还没有训练模型。下一章将训练模型，你可以使用经过训练的模型上的 KV 缓存（但是，KV 缓存仅用于推理期间）来生成连贯的文本。在这里，我们使用未经训练的模型来保持代码简单（更简单）。

更重要的是，`gpt_ch04.py` 和 `gpt_with_kv_cache.py` 实现都生成完全相同的文本。这告诉我们 KV 缓存已正确实现——很容易犯索引错误，从而导致不同的结果。


&nbsp;

## KV 缓存的优点和缺点

随着序列长度的增加，KV 缓存的优点和缺点在以下方面变得更加明显：

- [好] **计算效率提高**：如果不使用缓存，步骤 *t* 的注意力必须将新查询与 *t* 个先前的键进行比较，因此累积工作量呈二次方缩放，O(n²)。使用缓存，每个键和值计算一次，然后重用，将总的每步复杂度降低到线性，O(n)。

- [坏] **内存使用量线性增加**：每个新令牌都会追加到 KV 缓存中。对于长序列和较大的 LLM，累积的 KV 缓存会变得更大，这可能会消耗大量甚至令人望而却步的 (GPU) 内存。作为一种解决方法，我们可以截断 KV 缓存，但这会增加更多的复杂性（但这再次可能在部署 LLM 时非常值得。）



&nbsp;
## 优化 KV 缓存实现

虽然我在上面的 KV 缓存的概念实现有助于清晰度，并且主要面向代码可读性和教育目的，但在现实场景中部署它（特别是对于更大的模型和更长的序列长度）需要更仔细的优化。

&nbsp;
### 扩展缓存时的常见陷阱

- **内存碎片和重复分配**：如前所述，通过 `torch.cat` 连续连接张量会导致由于频繁的内存分配和重新分配而产生的性能瓶颈。

- **内存使用量的线性增长**：如果没有适当的处理，对于非常长的序列，KV 缓存大小变得不切实际。

&nbsp;
#### 技巧 1：预分配内存

与其重复连接张量，不如根据预期的最大序列长度预分配足够大的张量。这确保了一致的内存使用并减少了开销。在伪代码中，这可能如下所示：

```python
# 键和值的示例预分配
max_seq_len = 1024  # 最大预期序列长度
cache_k = torch.zeros((batch_size, num_heads, max_seq_len, head_dim), device=device)
cache_v = torch.zeros((batch_size, num_heads, max_seq_len, head_dim), device=device)
```

在推理期间，我们可以简单地写入这些预分配张量的切片。

&nbsp;
#### 技巧 2：通过滑动窗口截断缓存

为了避免耗尽我们的 GPU 内存，我们可以实现带有动态截断的滑动窗口方法。通过滑动窗口，我们在缓存中仅保留最后的 `window_size` 个令牌：


```python
# 滑动窗口缓存实现
window_size = 512
cache_k = cache_k[:, :, -window_size:, :]
cache_v = cache_v[:, :, -window_size:, :]
```

&nbsp;
#### 实践中的优化

你可以在 [`gpt_with_kv_cache_optimized.py`](gpt_with_kv_cache_optimized.py) 文件中找到这些优化。


在带有 M4 芯片 (CPU) 的 Mac Mini 上，对于 200 个令牌的生成和等于上下文长度的窗口大小（以保证相同的结果），代码运行时间比较如下：

|                                  | Tokens/sec |
| -------------------------------- | ---------- |
| `gpt_ch04.py`                    | 27         |
| `gpt_with_kv_cache.py`           | 144        |
| `gpt_with_kv_cache_optimized.py` | 166        |

不幸的是，速度优势在 CUDA 设备上消失了，因为这是一个微型模型，设备传输和通信超过了 KV 缓存对此小型模型的优势。


&nbsp;
## 其他资源

1. [Qwen3 从头开始的 KV 缓存基准测试](../../ch05/11_qwen3#pro-tip-2-speed-up-inference-with-compilation)
2. [Llama 3 从头开始的 KV 缓存基准测试](../../ch05/07_gpt_to_llama/README.md#pro-tip-3-speed-up-inference-with-compilation)
3. [从头开始理解和编码 LLM 中的 KV 缓存](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) -- 此 README 的更详细说明
