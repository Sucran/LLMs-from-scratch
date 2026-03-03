# 可选设置说明


本文档列出了设置你的机器和使用本仓库代码的不同方法。我建议从上到下浏览不同的部分，然后决定哪种方法最适合你的需求。

&nbsp;

## 快速开始

如果你已经在机器上安装了 Python，最快的入门方法是从代码仓库的根目录执行以下 pip 安装命令，安装 [../requirements.txt](../requirements.txt) 文件中的包依赖：

```bash
pip install -r requirements.txt
```

<br>

> **注意：** 如果你在 Google Colab 上运行任何笔记本并想安装依赖项，只需在笔记本顶部的代码单元格中运行以下代码：
> `pip install uv && uv pip install --system -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt`
> 或者，在克隆仓库后，你可以从项目根目录使用 `uv pip install --group bonus` 安装所有奖励材料的依赖项。这在你以后查看可选的奖励材料时不想单独安装它们时很有用。



在下面的视频中，我分享了我个人在电脑上设置 Python 环境的方法：

<br>
<br>

[![Link to the video](https://img.youtube.com/vi/yAcWnfsZhzo/0.jpg)](https://www.youtube.com/watch?v=yAcWnfsZhzo)


&nbsp;
# 本地设置

本节提供了在本地运行本书代码的建议。请注意，本书主要章节中的代码旨在在传统笔记本电脑上在合理的时间内运行，不需要专门的硬件。我在 M3 MacBook Air 笔记本电脑上测试了所有主要章节。此外，如果你的笔记本电脑或台式机有 NVIDIA GPU，代码将自动利用它。

&nbsp;
## 设置 Python

如果你还没有在机器上设置 Python，我在以下目录中写了我个人的 Python 设置偏好：

- [01_optional-python-setup-preferences](./01_optional-python-setup-preferences)
- [02_installing-python-libraries](./02_installing-python-libraries)

下面的 *使用 DevContainers* 部分概述了在你的机器上安装项目依赖项的另一种方法。

&nbsp;

## 使用 Docker DevContainers

作为上面 *设置 Python* 部分的替代方案，如果你更喜欢隔离项目依赖项和配置的开发设置，使用 Docker 是一个非常有效的解决方案。这种方法消除了手动安装软件包和库的需要，并确保了一致的开发环境。你可以找到更多关于设置 Docker 和使用 DevContainer 的说明：

- [03_optional-docker-environment](03_optional-docker-environment)

&nbsp;

## Visual Studio Code 编辑器

有很多好的代码编辑器可供选择。我的首选是流行的开源 [Visual Studio Code (VSCode)](https://code.visualstudio.com) 编辑器，它可以轻松地通过许多有用的插件和扩展进行增强（有关更多信息，请参阅下面的 *VSCode 扩展* 部分）。macOS、Linux 和 Windows 的下载说明可以在 [VSCode 主网站](https://code.visualstudio.com) 上找到。

&nbsp;

## VSCode 扩展

如果你使用 Visual Studio Code (VSCode) 作为主要代码编辑器，你可以在 `.vscode` 子文件夹中找到推荐的扩展。这些扩展提供了对本仓库有帮助的增强功能和工具。

要安装这些扩展，请在 VSCode 中打开此 "setup" 文件夹（文件 -> 打开文件夹...），然后单击右下角弹出菜单中的 "安装" 按钮。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/vs-code-extensions.webp?1" alt="1" width="700">

或者，你可以将 `.vscode` 扩展文件夹移动到此 GitHub 仓库的根目录：

```bash
mv setup/.vscode ./
```

然后，每次你打开 `LLMs-from-scratch` 主文件夹时，VSCode 都会自动检查你的系统上是否已安装推荐的扩展。

&nbsp;

# 云资源

本节介绍了运行本书中代码的云替代方案。

虽然代码可以在没有专用 GPU 的传统笔记本电脑和台式机上运行，但具有 NVIDIA GPU 的云平台可以大大缩短代码的运行时间，尤其是在第 5 到 7 章中。

&nbsp;

## 使用 Lightning Studio

为了获得流畅的云端开发体验，我推荐 [Lightning AI Studio](https://lightning.ai/) 平台，它允许用户设置持久环境，并在云 CPU 和 GPU 上使用 VSCode 和 Jupyter Lab。

启动新的 Studio 后，你可以打开终端并执行以下设置步骤来克隆仓库并安装依赖项：

```bash
git clone https://github.com/rasbt/LLMs-from-scratch.git
cd LLMs-from-scratch
pip install -r requirements.txt
```

（与 Google Colab 相比，这些步骤只需执行一次，因为即使你在 CPU 和 GPU 机器之间切换，Lightning AI Studio 环境也是持久的。）

然后，导航到你要运行的 Python 脚本或 Jupyter Notebook。或者，你也可以轻松连接 GPU 来加速代码的运行时间，例如，当你在第 5 章中预训练 LLM 或在第 6 和 7 章中对其进行微调时。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/studio.webp" alt="1" width="700">

&nbsp;

## 使用 Google Colab

要在云端使用 Google Colab 环境，请前往 [https://colab.research.google.com/](https://colab.research.google.com/) 并从 GitHub 菜单打开相应的章节笔记本，或将笔记本拖入 *上传* 字段，如下图所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_1.webp" alt="1" width="700">


还要确保你也上传了相关文件（笔记本从中导入的数据集文件和 .py 文件）到 Colab 环境，如下所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_2.webp" alt="2" width="700">


你可以选择通过更改 *运行时* 来在 GPU 上运行代码，如下图所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_3.webp" alt="3" width="700">


&nbsp;

# 有问题？

如果你有任何问题，请随时通过此 GitHub 仓库中的 [Discussions](https://github.com/rasbt/LLMs-from-scratch/discussions) 论坛联系。
