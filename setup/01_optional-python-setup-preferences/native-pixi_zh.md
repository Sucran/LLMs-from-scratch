# 原生 pixi Python 和包管理

本教程是 [`./native-uv.md`](native-uv.md) 文档的替代方案，适用于那些更喜欢 `pixi` 的原生命令而不是像 `conda` 和 `pip` 这样的传统环境和包管理器的用户。

请注意，pixi 在底层使用 `uv add`，如 [`./native-uv.md`](native-uv.md) 中所述。

Pixi 和 uv 都是用于 Python 的现代包和环境管理工具，但 pixi 是一个多语言包管理器，旨在不仅管理 Python，还管理其他语言（类似于 conda），而 uv 是一个特定于 Python 的工具，针对超快速依赖项解析和包安装进行了优化。

如果有人需要支持多种语言（不仅仅是 Python）的多语言包管理器，或者喜欢类似于 conda 的声明式环境管理方法，他们可能会选择 pixi 而不是 uv。有关更多信息，请访问官方 [pixi 文档](https://pixi.sh/latest/)。

在本教程中，我使用的是运行 macOS 的计算机，但此工作流程对于 Linux 机器是类似的，并且可能也适用于其他操作系统。

&nbsp;
## 1. 安装 pixi

根据你的操作系统，可以如下安装 Pixi。

<br>

**macOS 和 Linux**

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

或

```bash
wget -qO- https://pixi.sh/install.sh | sh
```

<br>

**Windows**

从官方 [文档](https://pixi.sh/latest/installation/#__tabbed_1_2) 下载安装程序或运行列出的 PowerShell 命令。



> **注意：**
> 有关更多安装选项，请参阅官方 [pixi 文档](https://pixi.sh/latest/)。


&nbsp;
## 1. 安装 Python

你可以使用 pixi 安装 Python：

```bash
pixi add python=3.10
```

> **注意：**
> 我建议安装比最新版本至少旧 2 个版本的 Python 版本，以确保 PyTorch 兼容性。例如，如果最新版本是 Python 3.13，我建议安装版本 3.10 或 3.11。你可以通过访问 [python.org](https://www.python.org) 找出最新的 Python 版本。

&nbsp;
## 3. 安装 Python 包和依赖项

要从 `pixi.toml` 文件（例如位于此 GitHub 仓库顶层的那个）安装所有必需的包，请运行以下命令，假设该文件与你的终端会话位于同一目录中：

```bash
pixi install
```

> **注意：**
> 如果你遇到依赖项问题（例如，如果你使用的是 Windows），你总是可以回退到 pip：`pixi run pip install -U -r requirements.txt`

默认情况下，`pixi install` 将创建一个特定于项目的单独虚拟环境。

你可以通过 `pixi add` 安装 `pixi.toml` 中未指定的新包，例如：

```bash
pixi add packaging
```

你可以通过 `pixi remove` 删除包，例如，

```bash
pixi remove packaging
```

&nbsp;
## 4. 运行 Python 代码

你的环境现在应该准备好运行仓库中的代码了。

或者，你可以通过运行此仓库中的 `python_environment_check.py` 脚本来运行环境检查：

```bash
pixi run python setup/02_installing-python-libraries/python_environment_check.py
```

<br>

**启动 JupyterLab**

你可以通过以下方式启动 JupyterLab 实例：

```bash
pixi run jupyter lab
```


---

有任何问题吗？请随时在 [Discussion Forum](https://github.com/rasbt/LLMs-from-scratch/discussions) 中联系。
