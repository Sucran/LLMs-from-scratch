# 从头开始的 Tiny Aya 3.35B

Tiny Aya 是 Cohere 推出的一个新的“小型”LLM，据称是 3B 参数规模级别中“最有能力的多语言开放权重模型”。（根据 [公告文章](https://cohere.com/blog/cohere-labs-tiny-aya)，Tiny Aya 优于 Qwen3-4B、Gemma 3 4B 和 Ministral 3 3B）。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/tiny-aya/01.webp">



这是一个在本地运行和试验的绝佳模型。唯一的警告是，虽然它是一个开放权重模型，但其许可条款相对受限，仅允许非商业用途。

除此之外，Arya 是一个 3.35B 参数的模型，有多种版本可供个人和（非商业）研究使用：

  - [tiny-aya-base](https://huggingface.co/CohereLabs/tiny-aya-base) (基础模型)
  - [tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) (跨语言和地区最平衡；笔记本默认)
  - [tiny-aya-fire](https://huggingface.co/CohereLabs/tiny-aya-fire) (针对南亚语言优化)
  - [tiny-aya-water](https://huggingface.co/CohereLabs/tiny-aya-water) (针对欧洲和亚太地区语言优化)
  - [tiny-aya-earth](https://huggingface.co/CohereLabs/tiny-aya-earth) (针对西亚和非洲语言优化)



更具体地说，这是模型优化的语言列表：

| 地区             | 语言                                                         | 优化模型        |
| ---------------- | ------------------------------------------------------------ | --------------- |
| **亚太地区**     | 繁体中文、广东话、越南语、他加禄语、爪哇语、高棉语、泰语、缅甸语、马来语、韩语、老挝语、印尼语、简体中文、日语 | tiny-aya-water  |
| **非洲**         | 祖鲁语、阿姆哈拉语、豪萨语、伊博语、斯瓦希里语、科萨语、沃洛夫语、绍纳语、约鲁巴语、尼日利亚洋泾浜语、马达加斯加语 | tiny-aya-earth  |
| **南亚**         | 泰卢固语、马拉地语、孟加拉语、泰米尔语、印地语、旁遮普语、古吉拉特语、乌尔都语、尼泊尔语 | tiny-aya-fire   |
| **欧洲**         | 加泰罗尼亚语、加利西亚语、荷兰语、丹麦语、芬兰语、捷克语、葡萄牙语、法语、立陶宛语、斯洛伐克语、巴斯克语、英语、瑞典语、波兰语、西班牙语、斯洛文尼亚语、乌克兰语、希腊语、博克马尔语、罗马尼亚语、塞尔维亚语、德语、意大利语、俄语、爱尔兰语、匈牙利语、保加利亚语、克罗地亚语、爱沙尼亚语、拉脱维亚语、威尔士语 | tiny-aya-water  |
| **西亚**         | 阿拉伯语、马耳他语、土耳其语、希伯来语、波斯语                    | tiny-aya-earth  |


在架构方面，Tiny Aya 是一个经典的解码器风格 transformer，有一些值得注意的修改（除了像 SwiGLU 和分组查询注意力这样的明显修改）：

1. **并行 transformer 块。** 并行 transformer 块从相同的归一化输入计算注意力和 MLP，然后一步将两者添加到残差中。我认为这是为了减少层内的串行依赖性，以提高计算吞吐量。

2. **滑动窗口注意力。** 具体来说，它使用类似于 Arcee Trinity 和 Olmo 3 的 3:1 局部:全局比例。窗口大小也是 4096。此外，类似于 Arcee，滑动窗口层使用 RoPE，而全注意力层使用 NoPE。

3. **LayerNorm。** 大多数架构都转向 RMSNorm，因为它在计算上稍微便宜一些并且性能良好。Tiny Aya 保持更经典，使用了 LayerNorm 的修改版本（此处的实现类似于标准 LayerNorm，但没有移位，即偏置参数）。



&nbsp;
## 文件

[standalone-tiny-aya.ipynb](standalone-tiny-aya.ipynb) 是一个独立的 Jupyter 笔记本，实现了 Tiny Aya 架构并加载了预训练权重。


替代的 [standalone-tiny-aya-plus-kvcache.ipynb](standalone-tiny-aya-plus-kv-cache.ipynb) 笔记本添加了 KV 缓存以获得更好的运行时性能（但增加了更多的代码复杂性）。要了解有关 KV 缓存的更多信息，请参阅我的 [从头开始理解和编码 LLM 中的 KV 缓存](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) 文章。


<br>

要了解有关架构差异的更多信息并阅读与其他架构的比较，请参阅我的 [大型 LLM 架构比较：从 DeepSeek-V3 到 Kimi K2：现代 LLM 架构设计一览](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) 文章。
