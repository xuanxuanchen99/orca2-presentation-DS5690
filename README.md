# Orca2 Paper Presentation - DS5690
This repository has materials for the paper presentation in Generative AI Models in Theory and Practice. 
The paper is "Orca 2: Teaching Small Language Models How to Reason."

## Overview of Orca 2
### Introduction and Problem

Large models such as GPT-4 are excellent at reasoning tasks. However, they are expensive and hard to deploy. In comparison, small language models are cheaper, but they often struggle with complex reasoning.

This paper targets that gap: how can we teach smaller models to reason more like large models without making them larger?

### Goal

Orca 2 aims to show that with the right training data and techniques, a 13B model can perform at levels comparable to much larger models. 
The goal is to train models to think through problems.

### Why It Matters

This effort is important because it challenges the "bigger is always better" assumption. 
If small models can reason well, we can make AI cheaper, more accessible, and easier to deploy in real-world settings.

## Core Ideas and Contributions

### Reasoning Style Supervision: 

Orca 2 is a small model trained using outputs from GPT 4. Specific reasoning strategies are used, including:

#### Step-by-step reasoning

**Question 1: What is step-by-step reasoning? Why does it matter in Orca 2’s success?**

#### Recall-then-answer

#### Direct answering

#### Explain-then-answer

#### Tool-use imitation

### Prompt Erasing: 

After GPT 4 creates a response using a specific reasoning style, the system instruction is removed. This helps the small model learn the reasoning process on its own instead of just copying the prompt wording.

### Smaller Model, Bigger Skillset: 

Orca 2-13B matches or beats LLaMA-2-Chat 70B and WizardLM 70B on reasoning benchmarks, in zero-shot settings.

### Focus on Reasoning, Not Dialogue: 

Unlike chat models, Orca 2 is not optimized for chit-chat. It is trained purely to think.

## Training Process with Pseudocode


    for task in dataset:
    
    # Step 1: Choose the right reasoning strategy for this task
    reasoning_type = assign_reasoning_strategy(task)

    # Step 2: Use GPT-4 to generate a response using that strategy
    teacher_output = generate_with_gpt4(task, reasoning_type)

    # Step 3: Remove the instruction prompt so the student must learn the pattern
    prompt_erased = remove_system_instruction(teacher_output)

    # Step 4: Train the Orca 2 model on the cleaned-up input/output pair
    student_model.train(prompt_erased)

**Explanation**

-The model sees a task such as a math problem or science question.

-It selects a reasoning strategy to apply.

-GPT-4 generates an answer using that strategy.

-The system prompt is erased, so the student model has to figure out the reasoning pattern on its own.

This process helps the model learn to choose the right way to solve each kind of problem.

## Results

Orca 2-13B beats models like LLaMA-2-70B and WizardLM-70B on 15 zero-shot reasoning benchmarks.

These include tasks from Big-Bench Hard, GSM8K, and MMLU.

In comparison, it underperforms on dialogue-style tasks like MT-Bench, as it is not tuned for chat.

## Critical Analysis

### Strengths

Matches or outperforms much larger models (like 70B) with just 13B parameters

Uses targeted reasoning strategies instead of one-size-fits-all instructions

Prompt Erasing improves generalization and prevents overfitting to GPT 4's wording

Encourages better transfer to unseen tasks

### Limitations

Still depends on GPT-4 outputs

**Question 2: What are the risks of relying on synthetic data from GPT 4?**

No RLHF (reinforcement learning with human feedback), so not ideal for safety-critical applications

### Future work

Add RLHF or dialog fine-tuning for broader use

Apply this method to multilingual or multimodal models

Explore use in domain-specific settings like medicine or law

## Real-World Impact

Makes AI cheaper, faster, and more energy-efficient

Enables small models to work in low-resource settings

Encourages future research into training strategies over brute-force scaling

## Code Demonstration

The jupyter notebook in this repo demonstrates the reasoning styles used to train Orca 2-7B which is available at

https://huggingface.co/microsoft/Orca-2-7b

## Resource Links

https://www.microsoft.com/en-us/research/blog/orca-2-teaching-small-language-models-how-to-reason/
Short Microsoft Research article to introduce Orca 2

https://arxiv.org/abs/2311.11045
Full research paper with details of Orca 2

https://huggingface.co/microsoft/Orca-2-7b
Smaller 7B parameter version of Orca 2

https://huggingface.co/microsoft/Orca-2-13b
Larger 13B parameter version of Orca 2

https://github.com/google-research/FLAN
An instruction-tuning dataset which is used in Orca 2’s training process

https://arxiv.org/abs/2307.09288
The LLaMA 2 family of models: the base architecture that Orca 2 builds on

## Citation

Microsoft Research. Orca 2: Teaching Small Language Models How to Reason. 2023.
https://arxiv.org/abs/2311.11045


## Thank you

Presented by Xuanxuan Chen


