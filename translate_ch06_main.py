# -*- coding: utf-8 -*-
import json
import os

# Translation dictionary
trans_map = {
    "Chapter 6: Finetuning for Text Classification": "第 6 章：用于文本分类的微调",
    "Supplementary code for the <a href=\"http://mng.bz/orYv\">Build a Large Language Model From Scratch</a> book by <a href=\"https://sebastianraschka.com\">Sebastian Raschka</a><br>": "本书 <a href=\"http://mng.bz/orYv\">Build a Large Language Model From Scratch</a> 的补充代码，作者 <a href=\"https://sebastianraschka.com\">Sebastian Raschka</a><br>",
    "Code repository: <a href=\"https://github.com/rasbt/LLMs-from-scratch\">https://github.com/rasbt/LLMs-from-scratch</a>": "代码仓库：<a href=\"https://github.com/rasbt/LLMs-from-scratch\">https://github.com/rasbt/LLMs-from-scratch</a>",
    "### 6.1 Different categories of finetuning": "### 6.1 微调的不同类别",
    "- No code in this section": "- 本节没有代码",
    "- The most common ways to finetune language models are instruction-finetuning and classification finetuning": "- 微调语言模型最常见的方法是指令微调（Instruction-Finetuning）和分类微调（Classification Finetuning）",
    "- Instruction-finetuning, depicted below, is the topic of the next chapter": "- 如下图所示的指令微调是下一章的主题",
    "- Classification finetuning, the topic of this chapter, is a procedure you may already be familiar with if you have a background in machine learning -- it's similar to training a convolutional network to classify handwritten digits, for example": "- 分类微调是本章的主题，如果您有机器学习背景，您可能已经熟悉这个过程——例如，它类似于训练卷积网络来分类手写数字",
    "- In classification finetuning, we have a specific number of class labels (for example, \"spam\" and \"not spam\") that the model can output": "- 在分类微调中，我们有特定数量的类别标签（例如，“垃圾邮件”和“非垃圾邮件”），模型可以输出这些标签",
    "- A classification finetuned model can only predict classes it has seen during training (for example, \"spam\" or \"not spam\"), whereas an instruction-finetuned model can usually perform many tasks": "- 分类微调后的模型只能预测它在训练期间见过的类别（例如，“垃圾邮件”或“非垃圾邮件”），而指令微调后的模型通常可以执行许多任务",
    "- We can think of a classification-finetuned model as a very specialized model; in practice, it is much easier to create a specialized model than a generalist model that performs well on many different tasks": "- 我们可以将分类微调后的模型视为一个非常专业的模型；在实践中，创建一个专业模型比创建一个在许多不同任务上表现良好的通用模型要容易得多",
    "### 6.2 Preparing the dataset": "### 6.2 准备数据集",
    "- This section prepares the dataset we use for classification finetuning": "- 本节准备我们用于分类微调的数据集",
    "- We use a dataset consisting of spam and non-spam text messages to finetune the LLM to classify them": "- 我们使用由垃圾邮件和非垃圾邮件文本消息组成的数据集来微调 LLM 以对其进行分类",
    "- First, we download and unzip the dataset": "- 首先，我们下载并解压数据集",
    "- The dataset is saved as a tab-separated text file, which we can load into a pandas DataFrame": "- 数据集保存为制表符分隔的文本文件，我们可以将其加载到 pandas DataFrame 中",
    "- When we check the class distribution, we see that the data contains \"ham\" (i.e., \"not spam\") much more frequently than \"spam\"": "- 当我们检查类别分布时，我们看到数据中包含“ham”（即“非垃圾邮件”）的频率远高于“spam”（垃圾邮件）",
    "- For simplicity, and because we prefer a small dataset for educational purposes anyway (it will make it possible to finetune the LLM faster), we subsample (undersample) the dataset so that it contains 747 instances from each class": "- 为了简单起见，而且因为我们要用于教学目的，我们也倾向于使用较小的数据集（这将使微调 LLM 更快），我们对数据集进行下采样（undersample），使其每个类别包含 747 个实例",
    "- (Next to undersampling, there are several other ways to deal with class balances, but they are out of the scope of a book on LLMs; you can find examples and more information in the [`imbalanced-learn` user guide](https://imbalanced-learn.org/stable/user_guide.html))": "- （除了下采样之外，还有其他几种处理类别平衡的方法，但它们超出了关于 LLM 的书籍的范围；您可以在 [`imbalanced-learn` 用户指南](https://imbalanced-learn.org/stable/user_guide.html) 中找到示例和更多信息）",
    "- Next, we change the string class labels \"ham\" and \"spam\" into integer class labels 0 and 1:": "- 接下来，我们将字符串类别标签“ham”和“spam”更改为整数类别标签 0 和 1：",
    "- Let's now define a function that randomly divides the dataset into training, validation, and test subsets": "- 现在让我们定义一个函数，将数据集随机划分为训练集、验证集和测试集",
    "### 6.3 Creating data loaders": "### 6.3 创建数据加载器",
    "- Note that the text messages have different lengths; if we want to combine multiple training examples in a batch, we have to either": "- 请注意，文本消息具有不同的长度；如果我们想在一个批次中组合多个训练示例，我们必须要么",
    "1. truncate all messages to the length of the shortest message in the dataset or batch": "1. 将所有消息截断为数据集或批次中最短消息的长度",
    "2. pad all messages to the length of the longest message in the dataset or batch": "2. 将所有消息填充到数据集或批次中最长消息的长度",
    "- We choose option 2 and pad all messages to the longest message in the dataset": "- 我们选择选项 2，并将所有消息填充到数据集中的最长消息",
    "- For that, we use `<|endoftext|>` as a padding token, as discussed in chapter 2": "- 为此，我们使用 `<|endoftext|>` 作为填充标记，如第 2 章所述",
    "- The `SpamDataset` class below identifies the longest sequence in the training dataset and adds the padding token to the others to match that sequence length": "- 下面的 `SpamDataset` 类识别训练数据集中最长的序列，并将填充标记添加到其他序列以匹配该序列长度",
    "- We also pad the validation and test set to the longest training sequence": "- 我们还将验证集和测试集填充到最长的训练序列长度",
    "- Note that validation and test set samples that are longer than the longest training example are being truncated via `encoded_text[:self.max_length]` in the `SpamDataset` code": "- 请注意，比最长训练示例更长的验证集和测试集样本将通过 `SpamDataset` 代码中的 `encoded_text[:self.max_length]` 被截断",
    "- This behavior is entirely optional, and it would also work well if we set `max_length=None` in both the validation and test set cases": "- 这种行为完全是可选的，如果我们在验证集和测试集情况下都设置 `max_length=None`，它也可以很好地工作",
    "- Next, we use the dataset to instantiate the data loaders, which is similar to creating the data loaders in previous chapters": "- 接下来，我们使用数据集实例化数据加载器，这类似于在前面章节中创建数据加载器",
    "- As a verification step, we iterate through the data loaders and ensure that the batches contain 8 training examples each, where each training example consists of 120 tokens": "- 作为验证步骤，我们迭代数据加载器并确保每个批次包含 8 个训练示例，其中每个训练示例由 120 个标记组成",
    "- Lastly, let's print the total number of batches in each dataset": "- 最后，让我们打印每个数据集中的总批次数",
    "### 6.4 Initializing a model with pretrained weights": "### 6.4 使用预训练权重初始化模型",
    "- In this section, we initialize the pretrained model we worked with in the previous chapter": "- 在本节中，我们初始化我们在上一章中使用的预训练模型",
    "- To ensure that the model was loaded correctly, let's double-check that it generates coherent text": "- 为了确保模型已正确加载，让我们仔细检查它是否生成连贯的文本",
    "- Before we finetune the model as a classifier, let's see if the model can perhaps already classify spam messages via prompting": "- 在我们将模型微调为分类器之前，让我们看看模型是否已经可以通过提示（prompting）对垃圾邮件进行分类",
    "- As we can see, the model is not very good at following instructions": "- 我们可以看到，该模型并不擅长遵循指令",
    "- This is expected, since it has only been pretrained and not instruction-finetuned (instruction finetuning will be covered in the next chapter)": "- 这是预料之中的，因为它只是经过了预训练，而没有经过指令微调（指令微调将在下一章介绍）",
    "### 6.5 Adding a classification head": "### 6.5 添加分类头",
    "- In this section, we are modifying the pretrained LLM to make it ready for classification finetuning": "- 在本节中，我们将修改预训练的 LLM，使其准备好进行分类微调",
    "- Let's take a look at the model architecture first": "- 首先让我们看看模型架构",
    "- Above, we can see the architecture we implemented in chapter 4 neatly laid out": "- 在上面，我们可以看到我们在第 4 章中实现的架构整齐地排列着",
    "- The goal is to replace and finetune the output layer": "- 我们的目标是替换并微调输出层",
    "- To achieve this, we first freeze the model, meaning that we make all layers non-trainable": "- 为了实现这一点，我们首先冻结模型，这意味着我们将所有层设置为不可训练",
    "- Then, we replace the output layer (`model.out_head`), which originally maps the layer inputs to 50,257 dimensions (the size of the vocabulary)": "- 然后，我们要替换输出层 (`model.out_head`)，该层最初将层输入映射到 50,257 个维度（词汇表的大小）",
    "- Since we finetune the model for binary classification (predicting 2 classes, \"spam\" and \"not spam\"), we can replace the output layer as shown below, which will be trainable by default": "- 由于我们要针对二元分类（预测 2 个类别，“垃圾邮件”和“非垃圾邮件”）对模型进行微调，我们可以如下所示替换输出层，该层默认情况下是可训练的",
    "- Note that we use `BASE_CONFIG[\"emb_dim\"]` (which is equal to 768 in the `\"gpt2-small (124M)\"` model) to keep the code below more general": "- 请注意，我们使用 `BASE_CONFIG[\"emb_dim\"]`（在 `\"gpt2-small (124M)\"` 模型中等于 768）以使下面的代码更通用",
    "- Technically, it's sufficient to only train the output layer": "- 从技术上讲，只训练输出层就足够了",
    "- However, as I found in [Finetuning Large Language Models](https://magazine.sebastianraschka.com/p/finetuning-large-language-models), experiments show that finetuning additional layers can noticeably improve the performance": "- 然而，正如我在 [微调大型语言模型](https://magazine.sebastianraschka.com/p/finetuning-large-language-models) 中发现的那样，实验表明微调额外的层可以显着提高性能",
    "- So, we are also making the last transformer block and the final `LayerNorm` module connecting the last transformer block to the output layer trainable": "- 因此，我们还将最后一个 transformer 块和连接最后一个 transformer 块与输出层的最终 `LayerNorm` 模块设置为可训练",
    "- We can still use this model similar to before in previous chapters": "- 我们仍然可以像前几章一样使用这个模型",
    "- For example, let's feed it some text input": "- 例如，让我们给它输入一些文本",
    "- What's different compared to previous chapters is that it now has two output dimensions instead of 50,257": "- 与前几章相比，不同之处在于它现在有两个输出维度，而不是 50,257",
    "- As discussed in previous chapters, for each input token, there's one output vector": "- 如前几章所述，对于每个输入标记，都有一个输出向量",
    "- Since we fed the model a text sample with 4 input tokens, the output consists of 4 2-dimensional output vectors above": "- 由于我们向模型输入了一个包含 4 个输入标记的文本样本，因此输出由上面的 4 个 2 维输出向量组成",
    "- In chapter 3, we discussed the attention mechanism, which connects each input token to each other input token": "- 在第 3 章中，我们讨论了注意力机制，它将每个输入标记连接到彼此的输入标记",
    "- In chapter 3, we then also introduced the causal attention mask that is used in GPT-like models; this causal mask lets a current token only attend to the current and previous token positions": "- 在第 3 章中，我们还介绍了 GPT 类模型中使用的因果注意力掩码；此因果掩码让当前标记仅关注当前和之前的标记位置",
    "- Based on this causal attention mechanism, the 4th (last) token contains the most information among all tokens because it's the only token that includes information about all other tokens": "- 基于这种因果注意力机制，第 4 个（最后一个）标记包含所有标记中最多的信息，因为它是唯一包含有关所有其他标记信息的标记",
    "- Hence, we are particularly interested in this last token, which we will finetune for the spam classification task": "- 因此，我们要特别关注这最后一个标记，我们将针对垃圾邮件分类任务对其进行微调",
    "### 6.6 Calculating the classification loss and accuracy": "### 6.6 计算分类损失和准确率",
    "- Before explaining the loss calculation, let's have a brief look at how the model outputs are turned into class labels": "- 在解释损失计算之前，让我们简要看看模型输出是如何转换为类别标签的",
    "- Similar to chapter 5, we convert the outputs (logits) into probability scores via the `softmax` function and then obtain the index position of the largest probability value via the `argmax` function": "- 与第 5 章类似，我们通过 `softmax` 函数将输出（logits）转换为概率分数，然后通过 `argmax` 函数获取最大概率值的索引位置",
    "- Note that the softmax function is optional here, as explained in chapter 5, because the largest outputs correspond to the largest probability scores": "- 请注意，这里的 softmax 函数是可选的，正如第 5 章所解释的那样，因为最大的输出对应于最大的概率分数",
    "- We can apply this concept to calculate the so-called classification accuracy, which computes the percentage of correct predictions in a given dataset": "- 我们可以应用这个概念来计算所谓的分类准确率，它计算给定数据集中正确预测的百分比",
    "- To calculate the classification accuracy, we can apply the preceding `argmax`-based prediction code to all examples in a dataset and calculate the fraction of correct predictions as follows:": "- 为了计算分类准确率，我们可以将前面的基于 `argmax` 的预测代码应用于数据集中的所有示例，并计算正确预测的比例，如下所示：",
    "- Let's apply the function to calculate the classification accuracies for the different datasets:": "- 让我们应用该函数来计算不同数据集的分类准确率：",
    "- As we can see, the prediction accuracies are not very good, since we haven't finetuned the model, yet": "- 我们可以看到，预测准确率不是很好，因为我们还没有微调模型",
    "- Before we can start finetuning (/training), we first have to define the loss function we want to optimize during training": "- 在我们开始微调（/训练）之前，我们首先必须定义我们在训练期间想要优化的损失函数",
    "- The goal is to maximize the spam classification accuracy of the model; however, classification accuracy is not a differentiable function": "- 目标是最大化模型的垃圾邮件分类准确率；然而，分类准确率不是一个可微函数",
    "- Hence, instead, we minimize the cross-entropy loss as a proxy for maximizing the classification accuracy (you can learn more about this topic in lecture 8 of my freely available [Introduction to Deep Learning](https://sebastianraschka.com/blog/2021/dl-course.html#l08-multinomial-logistic-regression--softmax-regression) class)": "- 因此，我们最小化交叉熵损失作为最大化分类准确率的代理（您可以在我免费提供的 [深度学习入门](https://sebastianraschka.com/blog/2021/dl-course.html#l08-multinomial-logistic-regression--softmax-regression) 课程的第 8 讲中了解有关此主题的更多信息）",
    "- The `calc_loss_batch` function is the same here as in chapter 5, except that we are only interested in optimizing the last token `model(input_batch)[:, -1, :]` instead of all tokens `model(input_batch)`": "- 这里的 `calc_loss_batch` 函数与第 5 章中的相同，只是我们只对优化最后一个标记 `model(input_batch)[:, -1, :]` 感兴趣，而不是所有标记 `model(input_batch)`",
    "The `calc_loss_loader` is exactly the same as in chapter 5": "`calc_loss_loader` 与第 5 章中的完全相同",
    "- Using the `calc_closs_loader`, we compute the initial training, validation, and test set losses before we start training": "- 使用 `calc_loss_loader`，我们在开始训练之前计算初始训练、验证和测试集损失",
    "- In the next section, we train the model to improve the loss values and consequently the classification accuracy": "- 在下一节中，我们将训练模型以改善损失值，从而提高分类准确率",
    "### 6.7 Finetuning the model on supervised data": "### 6.7 在监督数据上微调模型",
    "- In this section, we define and use the training function to improve the classification accuracy of the model": "- 在本节中，我们定义并使用训练函数来提高模型的分类准确率",
    "- The `train_classifier_simple` function below is practically the same as the `train_model_simple` function we used for pretraining the model in chapter 5": "- 下面的 `train_classifier_simple` 函数实际上与我们在第 5 章中用于预训练模型的 `train_model_simple` 函数相同",
    "- The only two differences are that we now \n  1. track the number of training examples seen (`examples_seen`) instead of the number of tokens seen\n  2. calculate the accuracy after each epoch instead of printing a sample text after each epoch": "- 唯一的两个区别是我们现在 \n  1. 跟踪看到的训练示例数量 (`examples_seen`) 而不是看到的标记数量\n  2. 在每个 epoch 后计算准确率，而不是在每个 epoch 后打印示例文本",
    "- The `evaluate_model` function used in the `train_classifier_simple` is the same as the one we used in chapter 5": "- `train_classifier_simple` 中使用的 `evaluate_model` 函数与我们在第 5 章中使用的相同",
    "- The training takes about 5 minutes on a M3 MacBook Air laptop computer and less than half a minute on a V100 or A100 GPU": "- 在 M3 MacBook Air 笔记本电脑上训练大约需要 5 分钟，在 V100 或 A100 GPU 上不到半分钟",
    "- Similar to chapter 5, we use matplotlib to plot the loss function for the training and validation set": "- 与第 5 章类似，我们使用 matplotlib 绘制训练集和验证集的损失函数",
    "- Above, based on the downward slope, we see that the model learns well": "- 在上面，基于向下的斜率，我们可以看到模型学习得很好",
    "- Furthermore, the fact that the training and validation loss are very close indicates that the model does not tend to overfit the training data": "- 此外，训练损失和验证损失非常接近的事实表明模型不倾向于过拟合训练数据",
    "- Similarly, we can plot the accuracy below": "- 同样，我们可以在下面绘制准确率",
    "- Based on the accuracy plot above, we can see that the model achieves a relatively high training and validation accuracy after epochs 4 and 5": "- 基于上面的准确率图，我们可以看到模型在 epoch 4 和 5 之后实现了相对较高的训练和验证准确率",
    "- However, we have to keep in mind that we specified `eval_iter=5` in the training function earlier, which means that we only estimated the training and validation set performances": "- 但是，我们要记住，我们之前在训练函数中指定了 `eval_iter=5`，这意味着我们只是估计了训练和验证集的性能",
    "- We can compute the training, validation, and test set performances over the complete dataset as follows below": "- 我们可以通过以下方式计算完整数据集上的训练、验证和测试集性能",
    "- We can see that the training and validation set performances are practically identical": "- 我们可以看到训练集和验证集的性能实际上是相同的",
    "- However, based on the slightly lower test set performance, we can see that the model overfits the training data to a very small degree, as well as the validation data that has been used for tweaking some of the hyperparameters, such as the learning rate": "- 然而，基于略低的测试集性能，我们可以看到模型在非常小的程度上过拟合了训练数据，以及用于调整某些超参数（如学习率）的验证数据",
    "- This is normal, however, and this gap could potentially be further reduced by increasing the model's dropout rate (`drop_rate`) or the `weight_decay` in the optimizer setting": "- 然而，这是正常的，并且可以通过增加模型的 dropout 率 (`drop_rate`) 或优化器设置中的 `weight_decay` 来进一步缩小这种差距",
    "### 6.8 Using the LLM as a spam classifier": "### 6.8 使用 LLM 作为垃圾邮件分类器",
    "- Finally, let's use the finetuned GPT model in action": "- 最后，让我们看看微调后的 GPT 模型的实际应用",
    "- The `classify_review` function below implements the data preprocessing steps similar to the `SpamDataset` we implemented earlier": "- 下面的 `classify_review` 函数实现了类似于我们之前实现的 `SpamDataset` 的数据预处理步骤",
    "- Then, the function returns the predicted integer class label from the model and returns the corresponding class name": "- 然后，该函数返回模型预测的整数类别标签并返回相应的类别名称",
    "- Let's try it out on a few examples below": "- 让我们在下面的一些示例中试一试",
    "- Finally, let's save the model in case we want to reuse the model later without having to train it again": "- 最后，让我们保存模型，以防我们以后想重用模型而不必再次训练它",
    "- Then, in a new session, we could load the model as follows": "- 然后，在新的会话中，我们可以按如下方式加载模型",
    "## Summary and takeaways": "## 总结和要点",
    "- See the [./gpt_class_finetune.py](./gpt_class_finetune.py) script, a self-contained script for classification finetuning": "- 参见 [./gpt_class_finetune.py](./gpt_class_finetune.py) 脚本，这是一个用于分类微调的独立脚本",
    "- You can find the exercise solutions in [./exercise-solutions.ipynb](./exercise-solutions.ipynb)": "- 您可以在 [./exercise-solutions.ipynb](./exercise-solutions.ipynb) 中找到练习解答",
    "- In addition, interested readers can find an introduction to parameter-efficient training with low-rank adaptation (LoRA) in [appendix E](../../appendix-E)": "- 此外，感兴趣的读者可以在 [附录 E](../../appendix-E) 中找到关于使用低秩适应 (LoRA) 进行参数高效训练的介绍",
    "### 6.1 Different categories of finetuning": "### 6.1 微调的不同类别",
    "- No code in this section": "- 本节没有代码",
    "- The most common ways to finetune language models are instruction-finetuning and classification finetuning\n- Instruction-finetuning, depicted below, is the topic of the next chapter": "- 微调语言模型最常见的方法是指令微调（Instruction-Finetuning）和分类微调（Classification Finetuning）\n- 如下图所示的指令微调是下一章的主题",
    "- Classification finetuning, the topic of this chapter, is a procedure you may already be familiar with if you have a background in machine learning -- it's similar to training a convolutional network to classify handwritten digits, for example\n- In classification finetuning, we have a specific number of class labels (for example, \"spam\" and \"not spam\") that the model can output\n- A classification finetuned model can only predict classes it has seen during training (for example, \"spam\" or \"not spam\"), whereas an instruction-finetuned model can usually perform many tasks\n- We can think of a classification-finetuned model as a very specialized model; in practice, it is much easier to create a specialized model than a generalist model that performs well on many different tasks": "- 分类微调是本章的主题，如果您有机器学习背景，您可能已经熟悉这个过程——例如，它类似于训练卷积网络来分类手写数字\n- 在分类微调中，我们有特定数量的类别标签（例如，“垃圾邮件”和“非垃圾邮件”），模型可以输出这些标签\n- 分类微调后的模型只能预测它在训练期间见过的类别（例如，“垃圾邮件”或“非垃圾邮件”），而指令微调后的模型通常可以执行许多任务\n- 我们可以将分类微调后的模型视为一个非常专业的模型；在实践中，创建一个专业模型比创建一个在许多不同任务上表现良好的通用模型要容易得多",
    "- This section prepares the dataset we use for classification finetuning\n- We use a dataset consisting of spam and non-spam text messages to finetune the LLM to classify them\n- First, we download and unzip the dataset": "- 本节准备我们用于分类微调的数据集\n- 我们使用由垃圾邮件和非垃圾邮件文本消息组成的数据集来微调 LLM 以对其进行分类\n- 首先，我们下载并解压数据集",
    "- The dataset is saved as a tab-separated text file, which we can load into a pandas DataFrame": "- 数据集保存为制表符分隔的文本文件，我们可以将其加载到 pandas DataFrame 中",
    "- For simplicity, and because we prefer a small dataset for educational purposes anyway (it will make it possible to finetune the LLM faster), we subsample (undersample) the dataset so that it contains 747 instances from each class\n- (Next to undersampling, there are several other ways to deal with class balances, but they are out of the scope of a book on LLMs; you can find examples and more information in the [`imbalanced-learn` user guide](https://imbalanced-learn.org/stable/user_guide.html))": "- 为了简单起见，而且因为我们要用于教学目的，我们也倾向于使用较小的数据集（这将使微调 LLM 更快），我们对数据集进行下采样（undersample），使其每个类别包含 747 个实例\n- （除了下采样之外，还有其他几种处理类别平衡的方法，但它们超出了关于 LLM 的书籍的范围；您可以在 [`imbalanced-learn` 用户指南](https://imbalanced-learn.org/stable/user_guide.html) 中找到示例和更多信息）",
    "- Note that the text messages have different lengths; if we want to combine multiple training examples in a batch, we have to either\n  1. truncate all messages to the length of the shortest message in the dataset or batch\n  2. pad all messages to the length of the longest message in the dataset or batch\n\n- We choose option 2 and pad all messages to the longest message in the dataset\n- For that, we use `<|endoftext|>` as a padding token, as discussed in chapter 2": "- 请注意，文本消息具有不同的长度；如果我们想在一个批次中组合多个训练示例，我们必须要么\n  1. 将所有消息截断为数据集或批次中最短消息的长度\n  2. 将所有消息填充到数据集或批次中最长消息的长度\n\n- 我们选择选项 2，并将所有消息填充到数据集中的最长消息\n- 为此，我们使用 `<|endoftext|>` 作为填充标记，如第 2 章所述",
    "- In this section, we define and use the training function to improve the classification accuracy of the model\n- The `train_classifier_simple` function below is practically the same as the `train_model_simple` function we used for pretraining the model in chapter 5\n- The only two differences are that we now \n  1. track the number of training examples seen (`examples_seen`) instead of the number of tokens seen\n  2. calculate the accuracy after each epoch instead of printing a sample text after each epoch": "- 在本节中，我们定义并使用训练函数来提高模型的分类准确率\n- 下面的 `train_classifier_simple` 函数实际上与我们在第 5 章中用于预训练模型的 `train_model_simple` 函数相同\n- 唯一的两个区别是我们现在 \n  1. 跟踪看到的训练示例数量 (`examples_seen`) 而不是看到的标记数量\n  2. 在每个 epoch 后计算准确率，而不是在每个 epoch 后打印示例文本",
}

def translate_notebook(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            new_source = []
            for line in cell['source']:
                translated_line = line
                # Try exact match first (stripping newline for matching, but keeping it for output)
                line_stripped = line.strip()
                if line_stripped in trans_map:
                    translated_line = trans_map[line_stripped]
                    if line.endswith('\n'):
                        translated_line += '\n'
                else:
                     # If no exact match, try to match parts if needed, or check if it's a multiline block in dict
                     # Join lines to check for multiline matches in the dictionary
                     pass
                new_source.append(translated_line)
            
            # Re-check for full cell content match (handling multiline strings in the dict)
            full_source = "".join(cell['source'])
            if full_source in trans_map:
                cell['source'] = [trans_map[full_source]]
            else:
                # Iterate again to apply single line translations if full source didn't match
                final_source = []
                for line in cell['source']:
                    line_content = line.rstrip('\n')
                    if line_content in trans_map:
                        trans_content = trans_map[line_content]
                        if line.endswith('\n'):
                            trans_content += '\n'
                        final_source.append(trans_content)
                    else:
                        final_source.append(line)
                cell['source'] = final_source

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

if __name__ == "__main__":
    input_file = "/Users/richard/Git/LLMs-from-scratch/ch06/01_main-chapter-code/ch06.ipynb"
    output_file = "/Users/richard/Git/LLMs-from-scratch/ch06/01_main-chapter-code/ch06_zh.ipynb"
    translate_notebook(input_file, output_file)
    print(f"Translated {input_file} to {output_file}")
