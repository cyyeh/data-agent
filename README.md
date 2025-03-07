# Data Agent

This repository contains the data agent code for the DABstep benchmark.

## Preparation

- install poetry
- prepare the environment: `poetry install`
- prepare the environment variables: copying `.env.example` to `.env` and filling in environment variables
  - `HF_TOKEN`: make sure to create a token with read access to the DABstep dataset, DABstep space and inference API
  - `OPENAI_API_KEY`: OpenAI API key if you want to use OpenAI model
  - `LANGFUSE_PUBLIC_KEY`: Langfuse public key
  - `LANGFUSE_SECRET_KEY`: Langfuse secret key
  - `LANGFUSE_HOST`: Langfuse host

## References

- [DABStep: Data Agent Benchmark for Multi-step Reasoning](https://huggingface.co/blog/dabstep)
- [Data Agent Benchmark for Multi-step Reasoning (DABstep) Dataset](https://huggingface.co/datasets/adyen/DABstep)
- [smolagents](https://huggingface.co/docs/smolagents/index)
- [🤗 AI Agents Course](https://huggingface.co/learn/agents-course)
- [Integrate Langfuse with smolagents](https://langfuse.com/docs/integrations/smolagents)