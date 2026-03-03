# 原生 uv Python 和包管理

本教程是 [README.md](./README.md) 文档中 *选项 1：使用 uv* 的替代方案，适用于那些更喜欢 `uv` 原生命令而不是 `uv pip` 接口的用户。虽然 `uv pip` 比纯 `pip` 更快，但 `uv` 的原生接口甚至比 `uv pip` 还要快，因为它的开销更小，并且不必处理对 PyPy 包依赖管理的旧支持。

下表比较了不同依赖项和包管理方法的速度。速度比较特别指的是安装期间的包依赖项解析，而不是已安装包的运行时性能。请注意，对于本项目而言，包安装是一次性过程，因此根据整体便利性而不仅仅是安装速度来选择首选方法是合理的。


| 命令                  | 速度比较 |
|-----------------------|-----------------|
| `conda install <pkg>` | 最慢（基准） |
| `pip install <pkg>`   | 比上面快 2-10 倍 |
| `uv pip install <pkg>`| 比上面快 5-10 倍 |
| `uv add <pkg>`        | 比上面快 2-5 倍 |

本教程重点介绍 `uv add`。


除此之外，与 [README.md](./README.md) 中的 *选项 1：使用 uv* 类似，本教程指导你使用 `uv` 进行 Python 设置和包安装过程。

在本教程中，我使用的是运行 macOS 的计算机，但此工作流程对于 Linux 机器是类似的，并且可能也适用于其他操作系统。


&nbsp;
## 1. 安装 uv

根据你的操作系统，可以如下安装 Uv。

<br>

**macOS 和 Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

或

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

<br>

**Windows**

```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | more"
```

&nbsp;

> **注意：**
> 有关更多安装选项，请参阅官方 [uv 文档](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)。

&nbsp;
## 2. 安装 Python 包和依赖项

要从 `pyproject.toml` 文件（例如位于此 GitHub 仓库顶层的那个）安装所有必需的包，请运行以下命令，假设该文件与你的终端会话位于同一目录中：

```bash
uv sync --dev --python 3.11
```

> **注意：**
> 如果你的系统上没有 Python 3.11，uv 将为你下载并安装它。
> 我建议使用比最新版本至少旧 1-3 个版本的 Python 版本，以确保 PyTorch 兼容性。例如，如果最新版本是 Python 3.13，我建议使用版本 3.10, 3.11, 3.12。你可以通过访问 [python.org](https://www.python.org/downloads/) 找出最新的 Python 版本。

> **注意：**
> 如果由于某些依赖项（例如，如果你使用的是 Windows）而在执行上述命令时遇到问题，你总是可以回退到常规 pip：
> `uv add pip`
> `uv run python -m pip install -U -r requirements.txt`


请注意，上面的 `uv sync` 命令将通过 `.venv` 子文件夹创建一个单独的虚拟环境。（如果你想删除虚拟环境以从头开始，只需删除 `.venv` 文件夹即可。）

你可以通过 `uv add` 安装 `pyproject.toml` 中未指定的新包，例如：

```bash
uv add packaging
```

你可以通过 `uv remove` 删除包，例如，

```bash
uv remove packaging
```



&nbsp;
## 3. 运行 Python 代码

<br>

你的环境现在应该准备好运行仓库中的代码了。

或者，你可以通过运行此仓库中的 `python_environment_check.py` 脚本来运行环境检查：

```bash
uv run python setup/02_installing-python-libraries/python_environment_check.py
```



<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/uv-run-check.png?1" width="700" height="auto" alt="Uv install">


<br>

**启动 JupyterLab**

你可以通过以下方式启动 JupyterLab 实例：

```bash
uv run jupyter lab
```

**跳过 `uv run` 命令**

如果你觉得输入 `uv run` 很麻烦，你可以按照如下所述手动激活虚拟环境。

在 macOS/Linux 上：

```bash
source .venv/bin/activate
```

在 Windows (PowerShell) 上：

```bash
.venv\Scripts\activate
```

然后，你可以通过以下方式运行脚本

```bash
python script.py
```

并通过以下方式启动 JupyterLab

```bash
jupyter lab
```

&nbsp;
> **注意：**
> 如果你遇到 jupyter lab 命令的问题，你也可以使用虚拟环境中的完整路径启动它。例如，在 Linux/macOS 上使用 `.venv/bin/jupyter lab`，或在 Windows 上使用 `.venv\Scripts\jupyter-lab`。

&nbsp;


&nbsp;

## 可选：手动管理虚拟环境

或者，你仍然可以使用 `uv pip install` 直接从仓库安装依赖项。但请注意，这不会像 `uv add` 那样将依赖项记录在 `uv.lock` 文件中。此外，它需要手动创建和激活虚拟环境：

<br>

**1. 创建一个新的虚拟环境**

运行以下命令手动创建一个新的虚拟环境，该环境将通过一个新的 `.venv` 子文件夹保存：

```bash
uv venv --python=python3.10
```

<br>

**2. 激活虚拟环境**

接下来，我们需要激活这个新的虚拟环境。

在 macOS/Linux 上：

```bash
source .venv/bin/activate
```

在 Windows (PowerShell) 上：

```bash
.venv\Scripts\activate
```

<br>

**3. 安装依赖项**

最后，我们可以使用 `uv pip` 接口从远程位置安装依赖项：

```bash
uv pip install -U -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt
```



---

有任何问题吗？请随时在 [Discussion Forum](https://github.com/rasbt/LLMs-from-scratch/discussions) 中联系。
