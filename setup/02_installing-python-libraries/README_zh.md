# 安装本书中使用的 Python 包和库

本文档提供了有关仔细检查已安装的 Python 版本和包的更多信息。（请参阅 [../01_optional-python-setup-preferences](../01_optional-python-setup-preferences) 文件夹以获取有关安装 Python 和 Python 包的更多信息。）

我在本书中使用了 [此处](https://github.com/rasbt/LLMs-from-scratch/blob/main/requirements.txt) 列出的库。这些库的较新版本很可能也兼容。但是，如果你在代码方面遇到任何问题，可以尝试使用这些库版本作为后备方案。



> **注意：**
> 如果你按照 [选项 1：使用 uv](../01_optional-python-setup-preferences/README.md) 中的描述使用 `uv`，你可以在下面的命令中通过 `uv pip` 替换 `pip`。例如，`pip install -r requirements.txt` 变为 `uv pip install -r requirements.txt`



为了最方便地安装这些要求，你可以使用此代码仓库根目录中的 `requirements.txt` 文件并执行以下命令：

```bash
pip install -r requirements.txt
```

或者，你可以通过 GitHub URL 安装它，如下所示：

```bash
pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/requirements.txt
```


然后，完成安装后，请使用以下命令检查所有包是否已安装且是最新的

```bash
python python_environment_check.py
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/check_1.jpg" width="600px">

还建议通过运行此目录中的 `python_environment_check.ipynb` 在 JupyterLab 中检查版本，理想情况下应该会得到与上面相同的结果。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/check_2.jpg" width="500px">

如果你看到以下问题，很可能是你的 JupyterLab 实例连接到了错误的 conda 环境：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/jupyter-issues.jpg" width="450px">

在这种情况下，你可能需要使用 `watermark` 检查你是否使用 `--conda` 标志在正确的 conda 环境中打开了 JupyterLab 实例：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/watermark.jpg" width="350px">


&nbsp;
## 安装 PyTorch

PyTorch 可以像任何其他 Python 库或包一样使用 pip 安装。例如：

```bash
pip install torch
```

但是，由于 PyTorch 是一个功能全面的库，具有兼容 CPU 和 GPU 的代码，因此安装可能需要额外的设置和说明（有关更多信息，请参阅书中的 *A.1.3 安装 PyTorch*）。

也强烈建议查阅 PyTorch 官方网站上的安装指南菜单：[https://pytorch.org](https://pytorch.org)。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/pytorch-installer.jpg" width="600px">

<br>



&nbsp;
## JupyterLab 技巧

如果你是在 JupyterLab 而不是 VSCode 中查看笔记本代码，请注意 JupyterLab（在其默认设置下）在最近的版本中存在滚动错误。我的建议是转到 Settings -> Settings Editor 并将 "Windowing mode" 更改为 "none"（如下图所示），这似乎可以解决该问题。


![Jupyter Glitch 1](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_1.webp)

<br>

![Jupyter Glitch 2](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_2.webp)

<br>

---




有任何问题吗？请随时在 [Discussion Forum](https://github.com/rasbt/LLMs-from-scratch/discussions) 中联系。
