# CoLLaMA: A Multilingual Instruction Dataset and Large Language Model for Code
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE) 
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

This is the repository for the `CoLLaMA` project, which aims to build a multilingual instruction dataset and large language model for coding tasks. 

## Overview
Current code instruction datasets, which are essential for instruction-tuning tasks, are often disorganized, monolingual, and single-programming language focused, while covering an insufficient variety of tasks. Open-source datasets for instruction tuning in coding tasks are also scarce.

For this end, we propose this project, with the following advantages:
- Multilingual Dataset: Our dataset incorporates code samples from a multitude of programming languages including Java, Python, C, C#, Go, PHP, JavaScript, and Ruby et.al. It also presents code instructions in both Chinese and English, enabling the model to learn in various programming language and spoken language contexts, and thereby enhancing its generalization ability.
- Task diversity: The dataset spans a broad range of coding tasks, such as code summarization, code generation, code search, and others. It incorporates tasks with varying complexities and requirements, from beginner to advanced levels. This comprehensive approach ensures our instructions can handle different types of coding tasks and covers a broad spectrum of programming skills and knowledge.
- Multi-programming paradigms: The project includes code examples from different programming paradigms, such as procedural, object-oriented, functional, and event-driven programming. This wide coverage provides the instruction-tuning model with a varied set of coding tasks to learn from and generate instructions for.
- Real-world code examples: The dataset incorporates code snippets or excerpts from actual projects or forums such as StackOverflow and Github, to present more realistic and practical coding tasks. This aids the instruction-tuning model in generating instructions applicable to real-world scenarios.
- Quality assurance: We are committed to providing an accurate and high-quality dataset for each coding task. For instance, the instruction dataset for code search, extracted from programming posts on Stackoverflow Q&A sites, is rigorously filtered and cleaned to ensure its usability in real Q&A applications.

The repository contains the following:
- The `MID_Dataset` used for fine-tuning the model
- The code for fine-tuning the model
- Model weight
- The code for evaluation

## Dataset release
[`data/MID_all_data.json`]() contains xx instruction-following data used for fine-tuning the CodeLLM model.
This file is a list of dictionaries, each dictionary contains the following fileds:
- `instruction`: describes the task that the model should perform. 
- `input`: optional code or context for the task. For example, if the instruction is 'Please summarize this PHP code.', the input is the PHP code.
- `output`: the answer to the instruction. 

All data in our collection is formatted into the same templates, where each sample is as follows:
```
[
{"instruction":  `string`,
"input":  `string`, # (may be empty)
"output": `string`}
]
```

Due to the different code tasks, we choose which filed to generate with  `gpt-3.5-turbo` or human. Unlike `self-struct` technology to generate data, most of the code in our data comes from the real world, whereas most instruction choices are generated by `gpt-3.5-turbo`. The detailed process of data processing is described in the next section.

## Dataset Collection & Processing
It includes 8 datasets for 8 diversited code tasks covering the following scenarios:

* **[code generation](data/code_generation)**: According to the natural languages input by the user, the corresponding code is generated.
* **[code summarization](data/code_summarization/)**: It aims to generate concise and readable summaries or description of source code. It involves automatically generating human-readable explations or summaries of code snippets, functions, or entire programs.
* **[code search](data/code_search/)**

    * **[code-to-code](data/code_search/code_to_code/)**

        * **[clone detection]()**: Given a piece of code, find another piece of code that is semantically related to it.
        * **[defect detection]()**: Given a source code, the task is to clarify what the specific defect of the code is. This include common errors such as null pointer, dereferences, array out of bounds, memory leaks, etc.
        * **[cloze test]()**: That involves completing or filling in the missing parts of a code snippet.
        * **[code repair]()**: It aims to automatically fix bugs in the code.
        * **[code translation]()**: Code translation refers to the process of converting source code from one programming language to another. It involves transforming the syntax, structure, and semantics of the original code while preserving its functionality and behavior.
    
    * **[query-to-code](data/code_search/query_to_code/)**: Given a natural language query and mutiple code snippets, the task is to search source code that its function matches the natural languag query.

A brief summary of [`MID_dataset`](data/MID_all_data.json) is given below:

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
        <td align="center">120k</td>  
        <td align="center">EN</td>  
        <td align="center">Go,Java,JavaScript,PHP,Python,Ruby</td>
     </tr>
     <tr>
        <td align="center">CodeSearchNet</td>
        <td align="center">120K</td>
        <td align="center">CN</td>
        <td align="center">Go,Java,JavaScript,PHP,Python,Ruby</td>
     </tr>
     <tr>
       <td colspan=3 rowspan=3 align="center">Code generation</td>  
        <td align="center">CodeSearchNet</td>  
        <td align="center">120k</td>  
        <td align="center">EN</td>  
        <td align="center">Go,Java,JavaScript,PHP,Python,Ruby</td>
     </tr>
     <tr>
        <td align="center">CodeGPT</td>
        <td align="center">29331</td>
        <td align="center">CN</td>
        <td align="center">C#,C,C++,Go,Java,JavaScript,PHP,Python,Ruby</td>
     </tr> 
        <td align="center">CodeSearchNet</td>  
        <td align="center">20k</td>  
        <td align="center">CN</td>  
        <td align="center">Go,Java,JavaScript,PHP,Python,Ruby</td>
     </tr>
     <tr>
        <td rowspan=7 align="center">Code Search</td>  
        <td rowspan=5 align="center">code-to-code</td>  
        <td align="center">Clone Detection</td>  
        <td align="center">BigCloneBench</td>
        <td align="center">20K</td>
        <td align="center">EN</td>  
        <td align="center">Java</td>
     </tr>
     <tr>
        <td align="center">Defect Detection</td>  
        <td align="center">Devign</td>  
        <td align="center">10101</td> 
        <td align="center">EN</td>   
        <td align="center">C</td>
     </tr>
     <tr>
        <td align="center">Cloze Test</td>  
        <td align="center">CT-all</td>  
        <td align="center">20K</td>  
        <td align="center">EN</td>  
        <td align="center">Go,Java,JavaScript,PHP,Python,Ruby</td>
     </tr>
     <tr>
        <td align="center">Code Repair</td>  
        <td align="center">Bug2Fix</td>  
        <td align="center">20K</td>  
        <td align="center">EN</td>  
        <td align="center">Java</td>
     </tr>
     <tr>
        <td align="center">Code Translation</td>  
        <td align="center">CodeTrans</td>  
        <td align="center">11749</td>  
        <td align="center">EN</td>  
        <td align="center">Java,C#</td>
     </tr>
     <tr>
        <td colspan=2 rowspan=2 align="center">query-to-code</td>  
        <td align="center">CodePro</td>  
        <td align="center">20222</td>  
        <td align="center">EN</td>  
        <td align="center">Python,SQL</td>
     </tr>
     <tr>
        <td align="center">CodePro</td>
        <td align="center">15234</td>
        <td align="center">CN</td>
        <td align="center">Python,SQL</td>
     </tr>
</table>

We mainly obtained datasets from [CodeSearchNet](https://github.com/github/CodeSearchNet),[CodeXGLUE](https://github.com/microsoft/CodeXGLUE), [codeGPT](https://github.com/zxx000728/CodeGPT) and [CodePro](https://github.com/hoogang/CodePro), processed them to obtain the aforementioned datasets, and concentrated them into one [dataset](data/MID_all_data.json).

## Finetuning

## Inference

## Citation
