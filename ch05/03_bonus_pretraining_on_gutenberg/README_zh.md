# 在 Project Gutenberg 数据集上预训练 GPT

此目录中的代码包含在 Project Gutenberg 提供的免费书籍上训练小型 GPT 模型的代码。

正如 Project Gutenberg 网站所述，“Project Gutenberg 的绝大多数电子书在美国都属于公有领域。”

请阅读 [Project Gutenberg 权限、许可和其他常见请求](https://www.gutenberg.org/policy/permission.html) 页面，了解有关使用 Project Gutenberg 提供的资源的更多信息。

&nbsp;
## 如何使用此代码

&nbsp;

### 1) 下载数据集

在本节中，我们使用 [`pgcorpus/gutenberg`](https://github.com/pgcorpus/gutenberg) GitHub 仓库中的代码从 Project Gutenberg 下载书籍。

截至撰写本文时，这将需要大约 50 GB 的磁盘空间，大约需要 10-15 小时，但这可能取决于自那时以来 Project Gutenberg 增长了多少。

&nbsp;
#### Linux 和 macOS 用户的下载说明


Linux 和 macOS 用户可以按照以下步骤下载数据集（如果你是 Windows 用户，请参阅下面的说明）：

1. 将 `03_bonus_pretraining_on_gutenberg` 文件夹设置为工作目录，以便在此文件夹中本地克隆 `gutenberg` 仓库（这是运行提供的脚本 `prepare_dataset.py` 和 `pretraining_simple.py` 所必需的）。例如，当在 `LLMs-from-scratch` 仓库的文件夹中时，通过以下方式导航到 *03_bonus_pretraining_on_gutenberg* 文件夹：
```bash
cd ch05/03_bonus_pretraining_on_gutenberg
```

2. 在那里克隆 `gutenberg` 仓库：
```bash
git clone https://github.com/pgcorpus/gutenberg.git
```

3. 导航到本地克隆的 `gutenberg` 仓库文件夹：
```bash
cd gutenberg
```

4. 从 `gutenberg` 仓库文件夹安装 *requirements.txt* 中定义的必需包：
```bash
pip install -r requirements.txt
```

5. 下载数据：
```bash
python get_data.py
```

6. 返回 `03_bonus_pretraining_on_gutenberg` 文件夹
```bash
cd ..
```

&nbsp;
#### Windows 用户的特别说明

[`pgcorpus/gutenberg`](https://github.com/pgcorpus/gutenberg) 代码与 Linux 和 macOS 兼容。但是，Windows 用户必须进行小的调整，例如将 `shell=True` 添加到 `subprocess` 调用并替换 `rsync`。

或者，在 Windows 上运行此代码的一种更简单的方法是使用“适用于 Linux 的 Windows 子系统” (WSL) 功能，该功能允许用户在 Windows 中使用 Ubuntu 运行 Linux 环境。有关更多信息，请阅读 [Microsoft 的官方安装说明](https://learn.microsoft.com/en-us/windows/wsl/install) 和 [教程](https://learn.microsoft.com/en-us/training/modules/wsl-introduction/)。

使用 WSL 时，请确保你已安装 Python 3（通过 `python3 --version` 检查，或例如使用 `sudo apt-get install -y python3.10` 安装 Python 3.10）并在那里安装以下包：

```bash
sudo apt-get update && \
sudo apt-get upgrade -y && \
sudo apt-get install -y python3-pip && \
sudo apt-get install -y python-is-python3 && \
sudo apt-get install -y rsync
```

> **注意：**
> 有关如何设置 Python 和安装包的说明，请参见 [可选 Python 设置首选项](../../setup/01_optional-python-setup-preferences/README.md) 和 [安装 Python 库](../../setup/02_installing-python-libraries/README.md)。
>
> 或者，此仓库提供了一个运行 Ubuntu 的 Docker 镜像。有关如何使用提供的 Docker 镜像运行容器的说明，请参见 [可选 Docker 环境](../../setup/03_optional-docker-environment/README.md)。

&nbsp;
### 2) 准备数据集

接下来，运行 `prepare_dataset.py` 脚本，该脚本将（截至撰写本文时为 60,173 个）文本文件连接成更少的较大文件，以便可以更有效地传输和访问它们：

```bash
python prepare_dataset.py \
  --data_dir gutenberg/data/raw \
  --max_size_mb 500 \
  --output_dir gutenberg_preprocessed
```

```
...
Skipping gutenberg/data/raw/PG29836_raw.txt as it does not contain primarily English text.                                     Skipping gutenberg/data/raw/PG16527_raw.txt as it does not contain primarily English text.                                     100%|██████████████████████████████████████████████████████████| 57250/57250 [25:04<00:00, 38.05it/s]
42 file(s) saved in /Users/sebastian/Developer/LLMs-from-scratch/ch05/03_bonus_pretraining_on_gutenberg/gutenberg_preprocessed
```


> **提示：**
> 请注意，为了简单起见，生成的文件以明文格式存储，并未进行预分词。但是，如果你计划更频繁地使用数据集或训练多个时期，你可能希望更新代码以将数据集存储为预分词形式以节省计算时间。有关更多信息，请参阅本页底部的 *设计决策和改进*。

> **提示：**
> 你可以选择较小的文件大小，例如 50 MB。这将导致更多的文件，但可能对于出于测试目的在少量文件上进行更快的预训练运行很有用。


&nbsp;
### 3) 运行预训练脚本

你可以如下运行预训练脚本。请注意，为了说明目的，显示的附加命令行参数带有默认值：

```bash
python pretraining_simple.py \
  --data_dir "gutenberg_preprocessed" \
  --n_epochs 1 \
  --batch_size 4 \
  --output_dir model_checkpoints
```

输出将按以下方式格式化：

> Total files: 3
> Tokenizing file 1 of 3: data_small/combined_1.txt
> Training ...
> Ep 1 (Step 0): Train loss 9.694, Val loss 9.724
> Ep 1 (Step 100): Train loss 6.672, Val loss 6.683
> Ep 1 (Step 200): Train loss 6.543, Val loss 6.434
> Ep 1 (Step 300): Train loss 5.772, Val loss 6.313
> Ep 1 (Step 400): Train loss 5.547, Val loss 6.249
> Ep 1 (Step 500): Train loss 6.182, Val loss 6.155
> Ep 1 (Step 600): Train loss 5.742, Val loss 6.122
> Ep 1 (Step 700): Train loss 6.309, Val loss 5.984
> Ep 1 (Step 800): Train loss 5.435, Val loss 5.975
> Ep 1 (Step 900): Train loss 5.582, Val loss 5.935
> ...
> Ep 1 (Step 31900): Train loss 3.664, Val loss 3.946
> Ep 1 (Step 32000): Train loss 3.493, Val loss 3.939
> Ep 1 (Step 32100): Train loss 3.940, Val loss 3.961
> Saved model_checkpoints/model_pg_32188.pth
> Book processed 3h 46m 55s
> Total time elapsed 3h 46m 55s
> ETA for remaining books: 7h 33m 50s
> Tokenizing file 2 of 3: data_small/combined_2.txt
> Training ...
> Ep 1 (Step 32200): Train loss 2.982, Val loss 4.094
> Ep 1 (Step 32300): Train loss 3.920, Val loss 4.097
> ...


&nbsp;
> **提示：**
> 在实践中，如果你使用的是 macOS 或 Linux，我建议使用 `tee` 命令除了在终端上打印日志输出外，还将日志输出保存到 `log.txt` 文件中：

```bash
python -u pretraining_simple.py | tee log.txt
```

&nbsp;
> **警告：**
> 请注意，在 V100 GPU 上，在 `gutenberg_preprocessed` 文件夹中的 1 个约 500 Mb 文本文件上进行训练大约需要 4 小时。
> 该文件夹包含 47 个文件，大约需要 200 小时（超过 1 周）才能完成。你可能希望在较少数量的文件上运行它。


&nbsp;
## 设计决策和改进

请注意，此代码侧重于保持简单和最小化以用于教育目的。代码可以通过以下方式进行改进，以提高建模性能和训练效率：

1. 修改 `prepare_dataset.py` 脚本以从每本书文件中剥离 Gutenberg 样板文本。
2. 更新数据准备和加载实用程序以预分词数据集并以分词形式保存，以便每次调用预训练脚本时不必重新分词。
3. 通过添加 [附录 D：为训练循环添加额外功能](../../appendix-D/01_main-chapter-code/appendix-D.ipynb) 中介绍的功能来更新 `train_model_simple` 脚本，即余弦衰减、线性预热和梯度裁剪。
4. 更新预训练脚本以保存优化器状态（参见第 5 章中的 *5.4 在 PyTorch 中加载和保存权重* 部分；[ch05.ipynb](../../ch05/01_main-chapter-code/ch05.ipynb)）并添加加载现有模型和优化器检查点并在训练运行中断时继续训练的选项。
5. 添加更高级的记录器（例如 Weights and Biases）以实时查看损失和验证曲线
6. 添加分布式数据并行 (DDP) 并在多个 GPU 上训练模型（参见附录 A 中的 *A.9.3 使用多个 GPU 训练* 部分；[DDP-script.py](../../appendix-A/01_main-chapter-code/DDP-script.py)）。
7. 将 `previous_chapter.py` 脚本中的从头开始 `MultiheadAttention` 类替换为 [高效多头注意力实现](../../ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb) 奖励部分中实现的更高效的 `MHAPyTorchScaledDotProduct` 类，该类通过 PyTorch 的 `nn.functional.scaled_dot_product_attention` 函数使用 Flash Attention。
8. 通过 [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) (`model = torch.compile`) 或 [thunder](https://github.com/Lightning-AI/lightning-thunder) (`model = thunder.jit(model)`) 优化模型来加速训练。
9. 实现梯度低秩投影 (GaLore) 以进一步加速预训练过程。这可以通过将 `AdamW` 优化器替换为 [GaLore Python 库](https://github.com/jiaweizzhao/GaLore) 中提供的 `GaLoreAdamW` 来实现。
