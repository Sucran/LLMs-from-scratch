# 从头开始带有聊天界面的 Qwen3



此奖励文件夹包含用于运行类似 ChatGPT 的用户界面以与预训练的 Qwen3 模型进行交互的代码。



![Chainlit UI example](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen3-chainlit.gif)



为了实现这个用户界面，我们使用开源的 [Chainlit Python 包](https://github.com/Chainlit/chainlit)。

&nbsp;
## 步骤 1：安装依赖项

首先，我们通过以下方式从 [requirements-extra.txt](requirements-extra.txt) 列表安装 `chainlit` 包和依赖项：

```bash
pip install -r requirements-extra.txt
```

或者，如果你使用的是 `uv`：

```bash
uv pip install -r requirements-extra.txt
```



&nbsp;

## 步骤 2：运行 `app` 代码

此文件夹包含 2 个文件：

1. [`qwen3-chat-interface.py`](qwen3-chat-interface.py)：此文件加载并使用处于思考模式的 Qwen3 0.6B 模型。
2. [`qwen3-chat-interface-multiturn.py`](qwen3-chat-interface-multiturn.py)：与上面相同，但配置为记住消息历史记录。

（打开并检查这些文件以了解更多信息。）

从终端运行以下命令之一以启动 UI 服务器：

```bash
chainlit run qwen3-chat-interface.py
```

或者，如果你使用的是 `uv`：

```bash
uv run chainlit run qwen3-chat-interface.py
```

运行上述命令之一应该会打开一个新的浏览器选项卡，你可以在其中与模型进行交互。如果浏览器选项卡没有自动打开，请检查终端命令并将本地地址复制到浏览器的地址栏中（通常，地址是 `http://localhost:8000`）。
