# 🤗🤗🤗课程讲义

宋曦轩 计算机系

# 课前准备

理论部分建议先修 [Mathematical Foundations of Machine Learning](https://oi-wiki.org/math/linear-algebra/)

实践部分建议了解 Python, Pytorch

## 软件准备：

建议使用 Linux 🐧 

安装以下 package:

[PyTorch](https://pytorch.org/)

[🤗transformers](https://huggingface.co/docs/transformers/installation)

```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simplepip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

```requirements.txt
protobuf
transformers
cpm_kernels
torch
gradio
mdtex2html
sentencepiece
accelerate
sse-starlette
streamlit
datasets
peft
tqdm
bitsandbytes
scipy
```

下载以下模型权重（可选）：

[gpt2 at main (huggingface.co)](https://huggingface.co/gpt2/tree/main)

[bert-base-chinese at main (huggingface.co)](https://huggingface.co/bert-base-chinese/tree/main)

[THUDM/chatglm2-6b-int4 at main (huggingface.co)](https://huggingface.co/THUDM/chatglm2-6b-int4/tree/main)

如果希望使用CPU推理chatglm，可以下载：

[THUDM/chatglm-6b-int4 at main (huggingface.co)](https://huggingface.co/THUDM/chatglm-6b-int4/tree/main)

## 硬件准备：

本次课程的实践中的部分内容需要GPU资源，如果缺少本地算力，可以尝试[Google Colab](https://colab.research.google.com/)，或者尝试使用CPU进行训练。

## 数据准备：

请提前下载：[openchat/openchat_sharegpt4_dataset at main (huggingface.co)](https://huggingface.co/datasets/openchat/openchat_sharegpt4_dataset/tree/main) 中的`sharegpt_gpt4.json`

## 思想准备（可选）：

阅读 [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

# 理论部分：自然语言处理

本部分内容对🤗并无实际帮助，如果您对本部分不感兴趣，可以直接跳转🤗。如果您在学习本内容时出现失语、语言障碍、[MASK]等症状，请减小学习率，并与身边同学及时沟通，以帮助恢复语言能力。

以下`1-4`小节介绍了 transformer 的原理，跳过本部分并不影响🤗。

## 1. 如何表示一个词的含义？ (Word Vector / Word Embedding)

## 2. 如何获取词在句子中的向量表示？ (Attention)

[An Apple a Day Keeps the Doctor Away](https://read.qxmd.com/read/25822137/association-between-apple-consumption-and-physician-visits-appealing-the-conventional-wisdom-that-an-apple-a-day-keeps-the-doctor-away)

一天一部 Iphone 让我与博士学位失之交臂？

[[1706.03762] Attention Is All You Need (arxiv.org)](https://arxiv.org/abs/1706.03762)

## 3. 如何在模型中存储知识🧀？(MLP)

[[2012.14913] Transformer Feed-Forward Layers Are Key-Value Memories (arxiv.org)](https://arxiv.org/abs/2012.14913)

## 4. 如何基于以上原理构建语言模型？(transformer)

[[1810.04805] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (arxiv.org)](https://arxiv.org/abs/1810.04805)

[Language Models are Unsupervised Multitask Learners (openai.com)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

[[2103.10360] GLM: General Language Model Pretraining with Autoregressive Blank Infilling (arxiv.org)](https://arxiv.org/abs/2103.10360)

以下`5-8`小节介绍了语言模型的训练方法，可以跳过理论讲解，在🤗🤗🤗的过程中🤗。

## 5. 如何训练语言模型？(Pretrain)

## 6. 如何使用语言模型完成具体下游任务？(Finetune)

## 7. 如何使用有限的硬件资源进行微调？(LoRA)

[[2106.09685] LoRA: Low-Rank Adaptation of Large Language Models (arxiv.org)](https://arxiv.org/abs/2106.09685)

## 8. 如何让语言模型更好的理解人类意图？(instruction tuning)

[arxiv.org/pdf/2203.02155.pdf](https://arxiv.org/pdf/2203.02155.pdf)

# 实践部分（课程作业）：🤗

## 0. Chat with LM (optional)

`ChatGLM.py`

和任意语言模型聊天 (ChatGPT, ChatGLM, Claude, Bard, Bing......) 。

例如询问：

```
什么情况下商贩会售卖淹死的鱼？
两杯50度的巧克力混合起来是多少度？
如何向盲人推销VR眼镜？
......
```

如果您发现了有趣的答案，请提交并在群聊中和大家分享。

这里列出了一些🤗上可以本地部署的模型：

[THUDM/chatglm-6b · Hugging Face](https://huggingface.co/THUDM/chatglm-6b)

[THUDM/chatglm2-6b · Hugging Face](https://huggingface.co/THUDM/chatglm2-6b)

[meta-llama/Llama-2-7b-chat-hf · Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

您也可以在🤗上寻找其他模型，或者使用您在`任务8`中训练的模型。

您还可以尝试和任意聊天框对话，比如 vscode 中的 [CodeGeeX](https://marketplace.visualstudio.com/items?itemName=aminer.codegeex) 代码生成插件。

## 1. A + B problem

`A+B_Problem.py`

请尝试将以下词的`Word Embedding`相加，并报告与所得向量最接近的词：

```
示例：
皇后-女人+男人=国王（答案在"学生", "骑士", "国王"中选择）
以下为题目：
僵尸+复活=？（答案在"鬼魂", "人类", "抱抱脸"中选择）
秀发-头发=？（答案在"闪亮", "昏暗", "电灯"中选择）
闪电-电流=？（答案在"快", "黄色", "下雨"中选择）
清华-北大=？（答案在"一流", "二流", "负一流"中选择）
大学+一流+海淀=？（答案在"清华", "北大", "斯坦福", "MIT"中选择）
欧盟-英国-法国-德国=？（答案在"俄罗斯", "欧盟", "波兰", "美国"中选择）
```

本次任务中，将使用 `bert-base-chinese` 的`Word Embedding`。

您也可以更换所使用的`Embedding`，并报告其他有趣的加式。

## 4. Manual mask filling

`gpt2/modeling_gpt2.py`

为了训练同学们的`coding`能力，参照`BERT`的预训练方式，讲师用自己小学期作业中[随机采样](https://simple.wikipedia.org/wiki/Infinite_monkey_theorem)的、[对比学习](https://arxiv.org/abs/2103.00020)的以及[自回归生成](https://codegeex.cn/playground)的代码制作了一个数据集，并用`[MASK]`对部分需要重构的代码片段进行了覆盖。同学们需要基于自身或者外部知识库，对`[MASK]`进行补全。

由于讲师不知道正确答案，因此讲师会通过[多数投票](https://arxiv.org/pdf/2305.18290.pdf)的方式选择最终`[MASK]`部分填写的代码。`Bonus`正比于讲师小学期的绩点。

由于本学期讲师没有小学期，所以会使用🤗上`gpt2-base`的代码作为本次作业的数据集，可以使用🤗作为外部知识库。

可以使用`Check_gpt2-base.py`进行检查（需下载模型至`model/gpt2-base`）。

提交填空后的`gpt2/modeling_gpt2.py`。

```
# 示例：
class MLP(nn.Module):
    def __init__(self, n_state, config):
        super().__init__()
        nx = config.n_embd  # FFN中间维度
        self.c_fc = [MASK](n_state, nx)
        self.c_proj = [MASK](nx, n_state)
        self.act = [MASK]  # 激活函数
        self.dropout = nn.Dropout(config.resid_pdrop)
    def forward(self, x):
        h = self.act([MASK])
        h2 = self.c_proj([MASK])
        return self.dropout(h2)
```

```
# 示例答案
class MLP(nn.Module):
    def __init__(self, n_state, config):
        super().__init__()
        nx = config.n_embd  # FFN中间维度
        self.c_fc = Linear(n_state, nx)
        self.c_proj = Linear(nx, n_state)
        self.act = gelu_new  # gpt2-base的激活函数
        self.dropout = nn.Dropout(config.resid_pdrop)
    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)
```

注：在🤗的实现中，🤗使用了Conv1D代替了Linear（也许是为了加速）。

## 5. Pretraining (optional)

预训练一个语言模型，结构不限(Encoder, Decoder, Encoder-Decoder)，数据不限，框架不限，算力不限。

以下是一些参考：

[GitHub - karpathy/minGPT: A minimal PyTorch re-implementation of the OpenAI GPT (Generative Pretrained Transformer) training](https://github.com/karpathy/minGPT)

[THUDM/SwissArmyTransformer: SwissArmyTransformer is a flexible and powerful library to develop your own Transformer variants. (github.com)](https://github.com/THUDM/SwissArmyTransformer)

[GitHub - microsoft/DeepSpeed: DeepSpeed is a deep learning optimization library that makes distributed training and inference easy, efficient, and effective.](https://github.com/microsoft/DeepSpeed)

[GitHub - NVIDIA/Megatron-LM: Ongoing research training transformer models at scale](https://github.com/NVIDIA/Megatron-LM)

本项作业无需提交。

## 8. Mean(GPT2, GPT4) = GPT3?

`Supervised_Fine-Tuning.py`

我们将使用 `GPT4` 生成的数据对 `GPT2` 进行 Supervised Fine-Tuning ，以此希望 `GPT2` 接近`GPT3` 的水平。

关于所使用的数据，更多信息参见：[imoneoi/openchat: OpenChat: Less is More for Open-source Models (github.com)](https://github.com/imoneoi/openchat)

为了减少显存资源占用，我们将在训练过程中使用LoRA和模型量化。

训练结束后，运行`ChatGPT2-base.py`进行对话🤗，您获得了GPT[MASK]？

提交任意对话记录，如微调中有任何问题，可一并提交。




