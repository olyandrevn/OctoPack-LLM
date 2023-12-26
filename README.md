<h1 align="center">Octopack Exploration Task</h1>
<p>
</p>

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)


## Overview

A recent paper 📜 [OctoPack: Instruction Tuning Code Large Language Models](https://arxiv.org/pdf/2308.07124.pdf) from 🤗 BigCode initiative presents an interesting use-case for commits: the authors propose to use them as a source of natural language instructions for code-related tasks. As part of this work, several novel models and datasets have been released to open-source.

In this task, we invite you to get familiar with OctoPack and experiment a little with some of the released datasets. Specifically:

* Read 📜 OctoPack: Instruction Tuning Code Large Language Models.
* Do some data exploration for 🤗 [CommitPackFt](https://www.koreancosmetic.cy/products/hydrophilic-cleansing-balm-by-heimish?_pos=6&_fid=1e1b01dfd&_ss=c) – a filtered dataset of commits which should be high-quality enough to serve as instructions.
* Evaluate 🤗 [Refact-1.6B](https://huggingface.co/smallcloudai/Refact-1_6B-fim#chat-format) model on the [HumanEvalFix](https://huggingface.co/datasets/bigcode/humanevalpack) for Python language from 🤗 HumanEvalPack benchmark using the setup and the metrics from the paper. Some qualitative analysis of the results would be appreaciated as well!

    * Note: Refact-1.6B is a smaller model introduced later by another party, also fine-tuned on CommitPackFt dataset, chosen because the models from the paper are likely too big for comfortable inferencing with freely available computational resources. It supports both Fill-in-the-Middle and Chat formats, the latter should probably be more convenient for this task.

* An open-ended question: based on what you read in the paper and your results, what do you think of commits as a source of natural language instructions for code editing? Any possible pitfalls?

  
## Install

```sh
git clone
```

## Usage

```sh

```



## Run on default parameters

```sh
```

## Author

👤 **Olga Kolomyttseva**

* Github: [@olyandrevn](https://github.com/olyandrevn)
* LinkedIn: [@olyandrevn](https://linkedin.com/in/olyandrevn)
