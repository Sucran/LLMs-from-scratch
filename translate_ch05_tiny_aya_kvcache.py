# -*- coding: utf-8 -*-
import json

# Input and output file paths
input_file = "ch05/15_tiny-aya/standalone-tiny-aya-plus-kv-cache.ipynb"
output_file = "ch05/15_tiny-aya/standalone-tiny-aya-plus-kv-cache_zh.ipynb"

# Translation mapping
translations = {
    "Supplementary code for the <a href=\"http://mng.bz/orYv\">Build a Large Language Model From Scratch</a> book by <a href=\"https://sebastianraschka.com\">Sebastian Raschka</a><br>": "Sebastian Raschka 的 <a href=\"http://mng.bz/orYv\">Build a Large Language Model From Scratch</a> 书籍的补充代码<br>",
    "<br>Code repository: <a href=\"https://github.com/rasbt/LLMs-from-scratch\">https://github.com/rasbt/LLMs-from-scratch</a>": "<br>代码仓库: <a href=\"https://github.com/rasbt/LLMs-from-scratch\">https://github.com/rasbt/LLMs-from-scratch</a>",
    "# Tiny Aya From Scratch (A Standalone Notebook)": "# 从头开始实现 Tiny Aya (独立笔记本)",
    "- This notebook is purposefully minimal and focuses on the code to re-implement Tiny Aya (3.35B) models from Cohere in pure PyTorch without relying on other external LLM libraries; Tiny Aya is interesting because it is a small but strong model with good multi-lingual support": "- 本笔记本特意保持极简，专注于在纯 PyTorch 中重新实现 Cohere 的 Tiny Aya (3.35B) 模型代码，而不依赖其他外部 LLM 库；Tiny Aya 很有趣，因为它是一个小巧但强大的模型，具有良好的多语言支持",
    "- For more information, see the official [Tiny Aya announcement](https://cohere.com/blog/cohere-labs-tiny-aya) and model cards:": "- 有关更多信息，请参阅官方 [Tiny Aya 公告](https://cohere.com/blog/cohere-labs-tiny-aya) 和模型卡：",
    "- Below is a table with more details regarding the language specialization (taken from their announcement blog post linked above)": "- 下面是有关语言专业化的更多详细信息的表格（摘自上面链接的公告博客文章）",
    "| Region        | Languages | Optimized Model |": "| 地区 | 语言 | 优化模型 |",
    "| **Asia Pacific** | Traditional Chinese, Cantonese, Vietnamese, Tagalog, Javanese, Khmer, Thai, Burmese, Malay, Korean, Lao, Indonesian, Simplified Chinese, Japanese | tiny-aya-water |": "| **亚太地区** | 繁体中文、粤语、越南语、他加禄语、爪哇语、高棉语、泰语、缅甸语、马来语、韩语、老挝语、印尼语、简体中文、日语 | tiny-aya-water |",
    "| **Africa** | Zulu, Amharic, Hausa, Igbo, Swahili, Xhosa, Wolof, Shona, Yoruba, Nigerian Pidgin, Malagasy | tiny-aya-earth |": "| **非洲** | 祖鲁语、阿姆哈拉语、豪萨语、伊博语、斯瓦希里语、科萨语、沃洛夫语、绍纳语、约鲁巴语、尼日利亚皮钦语、马尔加什语 | tiny-aya-earth |",
    "| **South Asia** | Telugu, Marathi, Bengali, Tamil, Hindi, Punjabi, Gujarati, Urdu, Nepali | tiny-aya-fire |": "| **南亚** | 泰卢固语、马拉地语、孟加拉语、泰米尔语、印地语、旁遮普语、古吉拉特语、乌尔都语、尼泊尔语 | tiny-aya-fire |",
    "| **Europe** | Catalan, Galician, Dutch, Danish, Finnish, Czech, Portuguese, French, Lithuanian, Slovak, Basque, English, Swedish, Polish, Spanish, Slovenian, Ukrainian, Greek, Bokmål, Romanian, Serbian, German, Italian, Russian, Irish, Hungarian, Bulgarian, Croatian, Estonian, Latvian, Welsh | tiny-aya-water |": "| **欧洲** | 加泰罗尼亚语、加利西亚语、荷兰语、丹麦语、芬兰语、捷克语、葡萄牙语、法语、立陶宛语、斯洛伐克语、巴斯克语、英语、瑞典语、波兰语、西班牙语、斯洛文尼亚语、乌克兰语、希腊语、博克马尔语、罗马尼亚语、塞尔维亚语、德语、意大利语、俄语、爱尔兰语、匈牙利语、保加利亚语、克罗地亚语、爱沙尼亚语、拉脱维亚语、威尔士语 | tiny-aya-water |",
    "| **West Asia** | Arabic, Maltese, Turkish, Hebrew, Persian | tiny-aya-earth |": "| **西亚** | 阿拉伯语、马耳他语、土耳其语、希伯来语、波斯语 | tiny-aya-earth |",
    "- Below is a side-by-side comparison with Qwen3 4B as a reference model; if you are interested in the Qwen3 standalone notebook, you can find it [here](../11_qwen3)": "- 下面是与 Qwen3 4B 作为参考模型的并排比较；如果您对 Qwen3 独立笔记本感兴趣，可以在 [这里](../11_qwen3) 找到它",
    "- About the code:": "- 关于代码：",
    "  - all code is my own code, mapping the Tiny Aya architecture onto the model code implemented in my [Build A Large Language Model (From Scratch)](http://mng.bz/orYv) book; the code is released under a permissive open-source Apache 2.0 license (see [LICENSE.txt](https://github.com/rasbt/LLMs-from-scratch/blob/main/LICENSE.txt))": "  - 所有代码均为我编写，将 Tiny Aya 架构映射到我的 [Build A Large Language Model (From Scratch)](http://mng.bz/orYv) 书中实现与模型代码上；代码在宽松的开源 Apache 2.0 许可下发布（请参阅 [LICENSE.txt](https://github.com/rasbt/LLMs-from-scratch/blob/main/LICENSE.txt)）",
    "# 1. Architecture code": "# 1. 架构代码",
    "# 2. Initialize model": "# 2. 初始化模型",
    "- The remainder of this notebook uses the Llama 3.2 1B model; to use the 3B model variant, just uncomment the second configuration file in the following code cell": "- 本笔记本的其余部分使用的是 Tiny Aya 模型；要使用其他变体，只需在以下代码单元中取消注释相应的配置文件",
    "# 3. Load tokenizer": "# 3. 加载分词器",
    "- Please note that Cohere requires that you accept the Tiny Aya licensing terms before you can download the files; to do this, you have to create a Hugging Face Hub account and visit the [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) repository to accept the terms": "- 请注意，Cohere 要求您在下载文件之前接受 Tiny Aya 许可条款；为此，您必须创建一个 Hugging Face Hub 帐户并访问 [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) 存储库以接受条款",
    "- Next, you will need to create an access token; to generate an access token with READ permissions, click on the profile picture in the upper right and click on \"Settings\"": "- 接下来，您需要创建一个访问令牌；要生成具有读取权限的访问令牌，请单击右上角的个人资料图片，然后单击“Settings”",
    "- Then, create and copy the access token so you can copy & paste it into the next code cell": "- 然后，创建并复制访问令牌，以便您可以将其复制并粘贴到下一个代码单元中",
    "- Note that if you use the fire, water, base, or earth model, you'd have to accept the licensing terms separately:": "- 请注意，如果您使用 fire、water、base 或 earth 模型，则必须单独接受许可条款：",
    "# Uncomment and run the following code if you are executing the notebook for the first time": "# 如果您是第一次执行笔记本，请取消注释并运行以下代码",
    "# 4. Load pretrained weights": "# 4. 加载预训练权重",
    "# 5. Generate text": "# 5. 生成文本",
    "# What's next?": "# 下一步是什么？",
    "- For those interested in a comprehensive guide on building a large language model from scratch and gaining a deeper understanding of its mechanics, you might like my [Build a Large Language Model (From Scratch)](http://mng.bz/orYv)": "- 对于那些有兴趣从头开始构建大型语言模型并深入了解其机制的综合指南的人，您可能会喜欢我的 [Build a Large Language Model (From Scratch)](http://mng.bz/orYv)",
}

def translate_notebook(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            new_source = []
            for line in cell['source']:
                translated_line = line
                for eng, chi in translations.items():
                    if eng in translated_line:
                        translated_line = translated_line.replace(eng, chi)
                new_source.append(translated_line)
            cell['source'] = new_source

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

if __name__ == "__main__":
    translate_notebook(input_file, output_file)
    print(f"Translated {input_file} to {output_file}")
