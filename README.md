# cs224n_LLM_agent

This repository contains the code for the Pong AI that uses Large Language Models (LLMs) as the policy and value function. The code is written in Python and uses the Trace (OptoPrime) library for optimization.

**Setup**

1. Create a `.env` file in the root directory of the repository with the following environment variables set:
   - `OAI_CONFIG_LIST`: path to the credential file for the LLMs
   - `OPENAI_API_KEY`: OPENAI API Key
   - `ANTHROPIC_API_KEY`: ANTHROPIC API Key
2. Create a credential file in the specified path with the following format (see [Trace AutoGen API Setup](https://github.com/microsoft/Trace?tab=readme-ov-file#using-autogen-as-backend) for more details):
   - `OAI_CONFIG_LIST` is a list of dictionaries with the following keys:
     - `model`: the model type (e.g. `gpt-4o-mini`)
     - `api_key`: the API key for the model
     

**Main Files**

* `simple_pong_ai.py`: A simple rule-based Pong AI agent
* `pong_LLM_agent.py`: The main script for training and evaluating the Pong AI using OptoPrime
