import datasets
import pandas as pd
from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel, FinalAnswerTool

from utils import download_dataset, setup_langfuse, run_benchmark, eval_accuracy

load_dotenv()


if __name__ == "__main__":
    tracer = setup_langfuse()
    context_files = download_dataset()

    models = {
        "gpt-4o-mini-2024-07-18": {
            "temperature": 0.0,
            "seed": 0,
            "max_completion_tokens": 4096,
        },
        "o3-mini-2025-01-31": {
            "seed": 0,
            "max_completion_tokens": 4096,
            "reasoning_effort": "low",
        },
    }
    MODEL_ID = "gpt-4o-mini-2024-07-18"
    MAX_STEPS = 7

    agent = CodeAgent(
        tools=[],
        model=OpenAIServerModel(
            MODEL_ID,
            **models[MODEL_ID]
        ),
        additional_authorized_imports=["numpy", "pandas", "json", "csv", "os", "glob", "markdown"],
        max_steps=MAX_STEPS,
    )
    # give agent power to open files
    agent.python_executor.send_tools({"open": open})

    SPLIT = "dev"
    # load dataset from Hub
    eval_dataset = datasets.load_dataset("adyen/DABstep", name="tasks", split=f"{SPLIT}")
    # eval_dataset = datasets.Dataset.from_list(list(filter(lambda x: x["level"] == 'easy', eval_dataset)))
    # eval_dataset = eval_dataset.select(range(3))

    # run agent
    agent_answers = run_benchmark(
        dataset=eval_dataset,
        agent=agent,
        context_files=context_files,
        tracer=tracer
    )

    # evaluation
    accuracy = eval_accuracy(
        agent_answers_df=pd.DataFrame(agent_answers),
        tasks_with_gt_df=eval_dataset.to_pandas(),
        save_eval_df=True,
    )

    print(f"Accuracy: {accuracy}")
