# RL Env Take-Home

## RL Environment Design — Multi-Step Investigation Task

### Overview

The goal of this project is to design and implement a **reinforcement learning environment for a multi-step reasoning task**.

Many modern AI systems are evaluated using **static benchmarks** (prompt → answer). However, real-world agent tasks often require **multi-step investigation**, where the model must gather information through actions before producing a final answer.

In this project, you will design an **interactive environment** where an agent must **investigate a software bug and determine its root cause**.

We are primarily evaluating:

- your understanding of **RL environment design**
- your ability to **define state, actions, and rewards**
- your thinking around **evaluation and reward shaping**

The coding portion should be reasonably straightforward— **using AI coding agents is not only allowed, but encouraged**. The important part is your environment design decisions.

---

# Task Description

You will build an environment where an agent must **identify the root cause of a bug in a small codebase**.

The agent will receive an initial description of a problem and must investigate the codebase by taking actions.

Example starting prompt:

```
Tests in this repository are failing due to incorrect output in the `calculate_discount` function.
Determine the root cause of the issue and propose the correct fix.
```

The agent must explore the repository and determine the cause of the bug before submitting an answer.

---

# Environment Requirements

Your environment should support **multi-step interaction** between the agent and the environment.

The agent should **not have direct access to the full codebase initially**. Instead, it must gather information through actions.

Your environment should expose a **set of actions** that the agent can use to investigate.

Examples (these are suggestions — you are free to design your own):

```
open_file(filename)
search(keyword)
view_test_output()
inspect_logs()
submit_answer()
```

Each action should return an observation from the environment.

The episode should end when:

- the agent submits an answer, or
- a maximum number of steps is reached.

---

# Environment Interface

Your environment should implement the following interface (or something similar):

```
reset()->observation
step(action)->observation,reward,done,info
```

Where:

- **observation** contains the information returned to the agent
- **reward** reflects the quality of the agent's behavior
- **done** indicates whether the episode has ended

You are free to design the structure of the observation.

---

# Codebase

You may create a small mock repository consisting of:

- 3–6 Python files
- a failing test case
- a bug in one function

The bug should require **investigation across multiple files**.

Example bug types:

- incorrect parameter handling
- logic error
- incorrect default value
- incorrect use of helper function

---

# Reward Design

You must design a reward function.

At minimum, your reward should account for:

### Correctness

Whether the agent identifies the correct root cause.

Example:

```
correct answer: +1
incorrect answer: -1
```

---

### Efficiency

Encourage agents to investigate efficiently.

Example:

```
step penalty: -0.01 per step
```

---

### Optional Additional Signals

You may include additional reward signals if you believe they improve the environment.

Examples:

- identifying the correct file
- referencing the correct function
- partial credit for identifying part of the issue

---

# Deliverables

Please submit the following:

## 1. Environment Implementation

Provide a runnable implementation of the environment.

The environment should include:

- environment class
- repository data
- reward logic
- example usage

---

## 2. Environment Design Document

Provide a short document (1–3 pages) explaining your design.

Include:

### Environment Design

Explain:

- state representation
- observation design
- action space
- termination conditions

---

### Reward Design

Explain:

- how rewards are computed
- why your reward design works
- potential reward hacking scenarios

---

### Failure Modes

Describe:

- how an agent could fail in this environment
- what behaviors this environment is meant to test

---

### Example Trajectory

Provide a short example interaction.

Example:

```
Step 1: search("discount")
Observation: function located in pricing.py

Step 2: open_file("pricing.py")
Observation: code snippet returned

Step 3: submit_answer("discount calculation ignores tax flag")
Reward: +1
Episode ends
```

---

# Expectations

We are not looking for a large or complex codebase.

Instead, we are looking for **thoughtful environment design**.

Strong submissions will demonstrate:

- clear state and action modeling
- thoughtful reward shaping
- understanding of multi-step reasoning environments
- awareness of potential reward exploitation
