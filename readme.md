# TravelPlanner-LLM

**Instruction-Tuned LLM for Structured Travel Itinerary Generation**

A lightweight **LLM fine-tuning project** that trains a language model to generate structured travel itineraries based on user instructions such as destination, budget, preferences, and constraints.

This project demonstrates a complete **LLM instruction-tuning pipeline**, including:

* dataset design
* LoRA fine-tuning
* structured prompt engineering
* inference pipeline
* automated evaluation
* **Base vs Fine-tuned model comparison**

---

# Example

User instruction

```text
帮我规划一个东京3日旅行，预算5000元，喜欢动漫和购物，不要太赶。
```

Fine-tuned model output

```text
### 行程概览
本次3日行程以动漫文化与购物体验为主，整体节奏轻松。

### Day 1
秋叶原动漫街区探索...

### Day 2
涩谷与原宿购物...

### Day 3
浅草寺与东京站周边散步...

### 预算建议
住宿约2500元，餐饮1200元...

### 注意事项
避免高峰时间购物区拥挤。
```

The fine-tuned model generates **consistent structured itineraries** aligned with user preferences.

---

# System Architecture

```text
User Instruction
        │
        ▼
Prompt Template
### Instruction
### Response
        │
        ▼
Base LLM (Qwen2.5-1.5B)
        │
LoRA Fine-Tuning
        │
        ▼
Fine-Tuned Travel Planner
        │
        ▼
Structured Travel Itinerary
```

---

# Training Pipeline

```text
Travel Planning Dataset
        │
        ▼
Instruction Formatting
### Instruction
### Response
        │
        ▼
LoRA Fine-Tuning
        │
        ▼
Fine-Tuned Model
        │
        ▼
Inference + Evaluation
```

---

# Project Structure

```text
travel-llm-finetune/
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── train.jsonl
│       ├── val.jsonl
│       └── test.jsonl
│
├── training/
│   └── train_lora.py
│
├── inference/
│   └── inference.py
│
├── evaluation/
│   ├── compare_base_vs_finetuned.py
│   └── results/
│
├── outputs/
│   └── qwen_travel_lora/
│
└── README.md
```

---

# Dataset

The dataset follows a **structured schema** designed specifically for travel planning tasks.

Example data sample

```json
{
"id": "travel_0001",
"task_type": "plan_generation",
"city": "新加坡",
"days": 3,
"budget_amount": 3000,
"budget_level": "medium",
"group_type": "solo",
"preferences": ["美食","城市观光"],
"constraints": ["不要太赶"],
"instruction": "帮我规划一个新加坡3日旅行...",
"output": "...",
"split": "train"
}
```

Dataset task types

| Task                  | Description                  |
| --------------------- | ---------------------------- |
| plan_generation       | Generate travel itineraries  |
| constraint_planning   | Follow travel constraints    |
| plan_revision         | Improve existing itineraries |
| structured_generation | Generate fixed-format output |

Dataset split

| Dataset    | Size |
| ---------- | ---- |
| Train      | 80   |
| Validation | 10   |
| Test       | 10   |

---

# Model

Base Model

```
Qwen2.5-1.5B-Instruct
```

Fine-tuning method

```
LoRA (Low-Rank Adaptation)
```

Why LoRA?

* memory efficient
* fast training
* widely used for LLM adaptation

---

# Training

Training uses **parameter-efficient LoRA fine-tuning**.

Key hyperparameters

| Parameter           | Value        |
| ------------------- | ------------ |
| Base model          | Qwen2.5-1.5B |
| Method              | LoRA         |
| Epochs              | 3            |
| Learning rate       | 2e-4         |
| Max sequence length | 1024         |

Run training

```bash
python training/train_lora.py
```

---

# Inference

Run the travel planner

```bash
python inference/inference.py
```

Example output

```text
### 行程概览
...

### Day 1
...

### Day 2
...

### Day 3
...

### 预算建议
...

### 注意事项
...
```

---

# Evaluation

The project includes an **automatic evaluation pipeline** comparing:

```
Base Model vs Fine-Tuned Model
```

Evaluation metrics

| Metric              | Description                   |
| ------------------- | ----------------------------- |
| Format Score        | Structured output correctness |
| Constraint Score    | Constraint following          |
| Preference Score    | Preference alignment          |
| Hallucination Score | Basic hallucination detection |
| Overall Score       | Weighted evaluation           |

Run evaluation

```bash
python evaluation/compare_base_vs_finetuned.py
```

---

# Results

Evaluation on held-out test set

| Metric              | Base Model | Fine-Tuned Model |
| ------------------- | ---------- | ---------------- |
| Overall Score       | 0.671      | **0.790**        |
| Format Score        | 0.766      | **0.929**        |
| Preference Score    | 0.650      | **0.800**        |
| Constraint Score    | 0.100      | 0.100            |
| Hallucination Score | 1.000      | 1.000            |

Key observations

* Fine-tuned model significantly improves **structured itinerary generation**
* Preference alignment improves after instruction tuning
* Constraint following remains challenging with limited data

---

# Example Comparison

Prompt

```
帮我规划一个东京3日旅行，预算5000元，喜欢动漫和购物。
```

Base model output

```
内容较随意，结构不稳定
```

Fine-tuned model output

```
### 行程概览
### Day 1
### Day 2
### Day 3
### 预算建议
### 注意事项
```

Fine-tuned model produces **consistent itinerary structure**.

---

# Limitations

Current limitations

* small training dataset
* simple rule-based evaluation
* limited constraint understanding

Future improvements

* larger dataset (500+ samples)
* LLM-as-judge evaluation
* better constraint-aware training

---

# Tech Stack

* Python
* PyTorch
* HuggingFace Transformers
* PEFT (LoRA)


---

# Key Takeaways

This project demonstrates a full pipeline for **LLM instruction tuning**:

* dataset construction
* LoRA fine-tuning
* structured generation
* automatic evaluation
* base vs fine-tuned comparison

Even with a small dataset, instruction tuning can significantly improve **structured output quality**.

---

# Author

Name: Diaobaodage00

Research interests

* LLM fine-tuning
* instruction tuning
* applied NLP systems

---

