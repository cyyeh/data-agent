# Data Agent

This repository contains the data agent code for the DABstep benchmark.

## Preparation

- install poetry
- prepare the environment: `poetry install`
- prepare the environment variables: copying `.env.example` to `.env` and filling in the `HF_TOKEN`(make sure to create a token with read access to the DABstep dataset, DABstep space and inference API)

## References

- [DABStep: Data Agent Benchmark for Multi-step Reasoning](https://huggingface.co/blog/dabstep)
- [Data Agent Benchmark for Multi-step Reasoning (DABstep) Dataset](https://huggingface.co/datasets/adyen/DABstep)
- [smolagents](https://huggingface.co/docs/smolagents/index)