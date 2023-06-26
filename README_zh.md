# CoLLaMA: A Multilingual Instruction Dataset and Large Language Model for Code

[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

\[ [English](README.md) | 中文 \]

`CoLLaMA` 项目旨在构建用于编码任务的多语言（中文和英文）指令调整数据集和大型语言模型。

<p align="center" width="100%">
<img src="https://i.postimg.cc/J7Ds1tw6/CoLLaMA.jpg"  width="40%" height="20%">
</p>

## 介绍

当前的代码指令数据集对于指令调整任务至关重要，但往往是杂乱无章、单语言和单一编程语言集中的，同时涵盖的任务种类也不够丰富。开源的编码任务指令调整数据集也很少。

为此，我们提出了`CoLLaMA`项目，其具有以下优势:

- `多语言数据集`:
  我们的数据集包含了多种编程语言的代码样本，包括Java、Python、C、C#、Go、PHP、JavaScript和Ruby等。它还提供了中英文两种语言的代码指令，使模型能够在不同的编程语言和口语语境中学习，从而增强其泛化能力。
- `任务多样性`: 数据集涵盖了广泛的编码任务，例如代码摘要、代码生成、代码搜索等。它包含了各种复杂度和要求的任务，从初学者到高级水平。这种全面的方法确保我们的指令可以处理不同类型的编码任务，并涵盖广泛的编程技能和知识。
- `多编程范例`: 该项目包括来自不同编程范式的代码示例，例如过程式、面向对象、函数式和事件驱动编程。这种广泛的覆盖范围为指令调整模型提供了各种不同的编码任务，以便学习和生成指令。
- `真实世界的代码示例`: 数据集包含来自实际项目或论坛（如StackOverflow和Github）的代码片段或摘录，以呈现更真实和实用的编码任务。这有助于指令调整模型生成适用于真实世界场景的指令。
- `质量保证`: 我们致力于为每个编码任务提供准确和高质量的数据集。例如，从Stackoverflow
  Q&A网站上提取的代码搜索指令数据集经过严格的过滤和清理，以确保其在实际的Q&A应用中可用。

改仓库包含以下内容:

- 用于微调模型的 `MulCo`
- 微调模型的代码
- 模型权重
- 评估代码

## 数据

`data/MID_all_data` 包含约88k条指令跟随数据，用于微调CoLLaMA模型。该文件是一个字典列表，每个字典包含以下字段：

- `instruction`: 描述模型应执行的任务。
- `input`: 任务的可选代码或上下文。例如，如果指令是“请总结这个PHP代码。”，则输入是PHP代码。
- `output`: 指令的答案。

我们收集的所有数据都采用相同的模板格式，每个样本如下：

```
[
{"instruction":  `string`,
"input":  `string`, # (may be empty)
"output": `string`}
]
```

由于代码任务的不同，我们使用 gpt-3.5-turbo
或人工生成数据。与自结构技术生成数据不同，我们的数据中大部分代码来自真实场景，而大部分指令选择则是由gpt-3.5-turbo生成的。数据处理的详细过程将在下一节中描述。

## 数据集收集与处理

它包括8个数据集，涵盖以下8个场景的不同编码任务：

* **[code generation](data/code_generation)**: 根据用户输入的自然语言，生成相应的代码。
* **[code summarization](data/code_summarization/)**: 生成简洁易读的源代码摘要或描述。它涉及自动生成代码片段、函数或整个程序的人类可读的解释或摘要。
* **[code search](data/code_search/)**

    * **[code-to-code](data/code_search/code_to_code/)**

        * **[clone detection]()**: 给定一段代码，找到另一段与其语义相关的代码。
        * **[defect detection]()**: 给定源代码，任务是澄清代码的具体缺陷是什么。这包括常见的错误，如空指针、解引用、数组越界、内存泄漏等。
        * **[code repair]()**: 涉及完成或填写代码片段中缺失的部分。
        * **[code repair]()**: 自动修复代码中的错误。
        * **[code translation]()**: 将源代码从一种编程语言转换为另一种编程语言的过程。它涉及将原始代码的语法、结构和语义进行转换，同时保留其功能和行为。

    * **[query-to-code](data/code_search/query_to_code/)**: 给定自然语言查询和多个代码片段，任务是搜索其功能与自然语言查询匹配的源代码。

`MulCo` 的简要概述如下:

<table border= "1" width= "600" align="center">
     <tr bgcolor="#D3D3D3">
        <td colspan=3 align="center">Task</td>  
        <td align="center">Source Dataset name</td>  
        <td align="center">Num</td>  
        <td align="center">Lang</td>  
        <td align="center">Programming Lang</td>
     </tr>
     <tr>
        <td colspan=3 rowspan=2 align="center">Code summarization</td>  
        <td align="center">CodeSearchNet</td>  
        <td align="center">10k</td>  
        <td align="center">EN</td>  
        <td align="center">Go,Java,JavaScript,PHP,Python,Ruby</td>
     </tr>
     <tr>
        <td align="center">CodeSearchNet</td>
        <td align="center">10K</td>
        <td align="center">CN</td>
        <td align="center">Go,Java,JavaScript,PHP,Python,Ruby</td>
     </tr>
     <tr>
       <td colspan=3 rowspan=3 align="center">Code generation</td>  
        <td align="center">CodeSearchNet</td>  
        <td align="center">10k</td>  
        <td align="center">EN</td>  
        <td align="center">Go,Java,JavaScript,PHP,Python,Ruby</td>
     </tr>
     <tr>
        <td align="center">CodeGPT</td>
        <td align="center">20k</td>
        <td align="center">CN</td>
        <td align="center">C#,C,C++,Go,Java,JavaScript,PHP,Python,Ruby</td>
     </tr> 
        <td align="center">CodeSearchNet</td>  
        <td align="center">5k</td>  
        <td align="center">CN</td>  
        <td align="center">Go,Java,JavaScript,PHP,Python,Ruby</td>
     </tr>
     <tr>
        <td rowspan=7 align="center">Code Search</td>  
        <td rowspan=5 align="center">code-to-code</td>  
        <td align="center">Clone Detection</td>  
        <td align="center">BigCloneBench</td>
        <td align="center">10k</td>
        <td align="center">EN</td>  
        <td align="center">Java</td>
     </tr>
     <tr>
        <td align="center">Defect Detection</td>  
        <td align="center">Devign</td>  
        <td align="center">5K</td> 
        <td align="center">EN</td>   
        <td align="center">C</td>
     </tr>
     <tr>
        <td align="center">Cloze Test</td>  
        <td align="center">CT-all</td>  
        <td align="center">5K</td>  
        <td align="center">EN</td>  
        <td align="center">Go,Java,JavaScript,PHP,Python,Ruby</td>
     </tr>
     <tr>
        <td align="center">Code Repair</td>  
        <td align="center">Bug2Fix</td>  
        <td align="center">5K</td>  
        <td align="center">EN</td>  
        <td align="center">Java</td>
     </tr>
     <tr>
        <td align="center">Code Translation</td>  
        <td align="center">CodeTrans</td>  
        <td align="center">5k</td>  
        <td align="center">EN</td>  
        <td align="center">Java,C#</td>
     </tr>
     <tr>
        <td colspan=2 rowspan=2 align="center">query-to-code</td>  
        <td align="center">CodePro</td>  
        <td align="center">10K</td>  
        <td align="center">EN</td>  
        <td align="center">Python,SQL</td>
     </tr>
     <tr>
        <td align="center">CodePro</td>
        <td align="center">5k</td>
        <td align="center">CN</td>
        <td align="center">Python,SQL</td>
     </tr>
</table>

我们主要从 [CodeSearchNet](https://github.com/github/CodeSearchNet),[CodeXGLUE](https://github.com/microsoft/CodeXGLUE),  [codeGPT](https://github.com/zxx000728/CodeGPT)
和 [CodePro](https://github.com/hoogang/CodePro)中获取数据, 并对它们进行处理以获取上述数据集,
最终合并到一个 [dataset](data/MID_all_data.json)中.

## Finetuning

微调过程基本上遵循[codealpace](https://github.com/sahil280114/codealpaca/tree/master).

要复制一个经过fine-tuning的LLaMA版本，请按照以下步骤操作:

为了有效地对llama-7b模型进行fine-tuning，我们使用了8个A100 80GB GPU。建议至少使用两个A100 80GB
GPU以避免内存不足。同时，您需要调整训练参数和deepspeed配置文件。

在fine-tuning之前，请确保先安装所有依赖项:

```bash
pip install -r requirements.txt
```

执行以下脚本，使用deepspeed在一台配备8个A100 80G GPU的机器上，对LLaMA-7B模型进行fine-tuning。

```bash
torchrun --nproc_per_node=8 --master_port=2000 train.py \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --data_path MID_train_EN_data.json \
    --fp16 True \
    --output_dir <output_path> \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --deepspeed ds_config.json
```

您可以将`--data_path`替换为您自己的数据集。

您可以使用以下命令将fine-tuned模型上传至Huggingface（引用自[此处](https://github.com/minosvasilias/godot-dodo/tree/main)）：

```bash
python finetuning/push_to_hub.py --model_name_or_path PATH_TO_FINETUNED_MODEL/ --push_name HF_MODEL_NAME --auth_token HF_ACCESS_TOKEN
```

模型已于Huggingface开放:[CoLLaMA-7b](https://huggingface.co/DaliahX/CoLLaMA-7b/upload/main)

## 评估 (TODO)

## 引用

<div>
<div align="center">
    <a target='_blank'>Gang Hu<sup>1</sup></span>&emsp;
    <a target='_blank'>Xi Wen<sup>1</sup></span>&emsp;
    <a target='_blank'>Xin Liu<sup>1</sup></a>&emsp;
    <a href='https://jimin.chancefocus.com/' target='_blank'>Jimin Huang<sup>2</sup></a>&emsp;
    <a target='_blank'>Qianqian Xie*<sup>3</sup></a>&emsp;

</div>
<div>
<div align="center">
    <sup>1</sup>School of Computer Science, Yunnan University&emsp;
  <sup>2</sup>ChanceFocus AMC&emsp;
   <sup>3</sup>School of Computer Science, Wuhan University&emsp;
</div>

```
@misc{Hu2023CoLLaMA,
      title={CoLLaMA: A Multilingual Instruction Dataset and Large Language Model for Code}, 
      author={Gang Hu and Xi Wen and Xin Liu and Jimin Huang and Qianqian Xie},
      year={2023},
}
``` 

</div>
</div>