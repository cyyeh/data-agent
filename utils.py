import base64
import os
import shutil
import json
import uuid
import pandas as pd
from datetime import datetime
from pathlib import Path
from opentelemetry.trace import Tracer

import datasets
from huggingface_hub import hf_hub_download
from dabstep_benchmark.utils import evaluate
from smolagents import CodeAgent
from smolagents.agents import ActionStep, TaskStep, PlanningStep


CONTEXT_FILENAMES = [
    "data/context/acquirer_countries.csv",
    "data/context/payments-readme.md",
    "data/context/payments.csv",
    "data/context/merchant_category_codes.csv",
    "data/context/fees.json",
    "data/context/merchant_data.json",
    "data/context/manual.md",
]


PROMPT = """You are an expert data analyst and you will answer factoid questions by loading and referencing the files/documents listed below.
You have these files available:
{context_files}
Don't forget to reference any documentation in the data dir before answering a question.

Here is the question you need to answer:
{question}

Here are the guidelines you must follow when answering the question above:
{guidelines}
"""


def setup_langfuse() -> Tracer:
    from dotenv import load_dotenv
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from openinference.instrumentation.smolagents import SmolagentsInstrumentor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
 
    load_dotenv()

    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")

    LANGFUSE_AUTH=base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{LANGFUSE_HOST}/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

    trace_provider = TracerProvider()
    trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
    
    SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)

    trace.set_tracer_provider(trace_provider)
    return trace.get_tracer("my.tracer.name")


def download_dataset(data_destination_dir: str = "/tmp/DABstep-data"):
    if Path(data_destination_dir).exists():
        shutil.rmtree(data_destination_dir)

    for filename in CONTEXT_FILENAMES:
        hf_hub_download(
            repo_id="adyen/DABstep",
            repo_type="dataset",
            filename=filename,
            local_dir=data_destination_dir,
            force_download=True
        )

    context_files = [f"{data_destination_dir}/{filename}" for filename in CONTEXT_FILENAMES]

    for file in context_files:
        assert Path(file).exists(), f"{file} does not exist."

    return context_files


# You can inspect the steps taken by the agent by doing this
def clean_reasoning_trace(trace: list[ActionStep, TaskStep, PlanningStep]) -> list:
    for step in trace:
        # Remove memory from logs to make them more compact.
        if hasattr(step, "memory"):
            step.memory = None
        if isinstance(step, ActionStep):
            step.agent_memory = None
    return trace


def run_benchmark(
    dataset: datasets.Dataset, 
    agent: CodeAgent,
    context_files: list[str],
    tracer: Tracer
) -> list[dict]:
    session_id = str(uuid.uuid4())

    agent_answers = []
    for task in dataset:
        with tracer.start_as_current_span("Smolagent-Trace") as span:
            span.set_attribute("langfuse.session.id", session_id)
            tid = task['task_id']

            prompt = PROMPT.format(
                context_files=context_files,
                question=task['question'],
                guidelines=task['guidelines']
            )

            answer = agent.run(prompt)

            task_answer = {
                "trace_id": span.get_span_context().trace_id,
                "task_id": str(tid),
                "agent_answer": str(answer),
                "reasoning_trace": str(clean_reasoning_trace(agent.memory.steps))
            }

            agent_answers.append(task_answer)

    return agent_answers


def write_jsonl(data: list[dict], filepath: Path) -> None:
    """Write a list of dictionaries to a JSONL file."""
    # Ensure the directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as file:
        for entry in data:
            file.write(json.dumps(entry) + "\n")


def eval_accuracy(
    agent_answers_df: pd.DataFrame,
    tasks_with_gt_df: pd.DataFrame,
    save_eval_df: bool = False,
    eval_df_path: str = f"outputs/eval_df_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
) -> float:
    task_scores = evaluate(agent_answers=agent_answers_df, tasks_with_gt=tasks_with_gt_df)
    task_scores_df = pd.DataFrame(task_scores)
    task_scores_df["correct_answer"] = tasks_with_gt_df["answer"]
    task_scores_df["question"] = tasks_with_gt_df["question"]
    task_scores_df["trace_id"] = agent_answers_df["trace_id"]
    accuracy = task_scores_df["score"].mean()

    if save_eval_df:
        if not Path("outputs").exists():
            Path("outputs").mkdir(parents=True, exist_ok=True)

        task_scores_df.to_csv(eval_df_path, index=False)

    return accuracy
