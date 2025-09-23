# Learning Game-Playing Agents with Generative Code Optimization

This repository contains code for *Learning Game-Playing Agents with Generative Code Optimization* (ICML 2025 PRAL Workshop). We use Trace LLM optimizers (OptoPrime) to optimize python policies to play Atari games. Paper link: https://openreview.net/forum?id=ZM65X3NoTd

## Setup

Create a `.env` file in the root directory of the repository with the following environment variables set:
```
TRACE_CUSTOMLLM_MODEL=xxxxx
TRACE_CUSTOMLLM_URL=xxxxx
TRACE_CUSTOMLLM_API_KEY=xxxxx
TRACE_DEFAULT_LLM_BACKEND=xxxxx
```

## Pong AI

* `pong_ocatari_LLM_agent.py`: Primary script for training and evaluating the Pong AI with OCAtari API for environment interaction and Trace LLM optimizers for optimizing the policy.
* `evaluate_Pong_policy.py`: This script is used to load checkpoints and evaluate the performance of the trained Pong AI policy.

**Additional Scripts**
* `pong_LLM_agent.py`: Initial script for training and evaluating the Pong AI using the Gymnasium API alongside Trace LLM optimizers.
* `simple_pong_ai.py`: Implements a basic rule-based Pong AI agent as a simple baseline.
