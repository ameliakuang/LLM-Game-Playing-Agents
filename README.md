# cs224n_LLM_agent

This repository contains the code for the Pong AI that uses Large Language Models (LLMs) as the policy and value function. The code is written in Python and uses the Trace (OptoPrime) library for optimization.

**Setup**

1. Create a `.env` file in the root directory of the repository with the following environment variables set:
```
TRACE_CUSTOMLLM_MODEL=xxxxx
TRACE_CUSTOMLLM_URL=xxxxx
TRACE_CUSTOMLLM_API_KEY=xxxxx
TRACE_DEFAULT_LLM_BACKEND=xxxxx
```

**Main Files**

* `simple_pong_ai.py`: A simple rule-based Pong AI agent
* `pong_LLM_agent.py`: The main script for training and evaluating the Pong AI using Gymnasium API and OptoPrime
* `pong_ocatari_LLM_agent.py`: The main script for training and evaluating the Pong AI using OCAtari API and OptoPrime