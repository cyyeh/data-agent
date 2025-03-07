import base64
import os
import shutil

import datasets
from huggingface_hub import hf_hub_download
from smolagents import CodeAgent


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


def setup_langfuse():
    from dotenv import load_dotenv
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


def download_dataset(data_destination_dir: str = "/tmp/DABstep-data"):
    if os.path.exists(data_destination_dir):
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
        assert os.path.exists(file), f"{file} does not exist."

    return context_files


def run_benchmark(dataset: datasets.Dataset, agent: CodeAgent, context_files: list[str]) -> list[dict]:
    global PROMPT
    
    agent_answers = []
    for task in dataset:
        tid = task['task_id']

        prompt = PROMPT.format(
            context_files=context_files,
            question=task['question'],
            guidelines=task['guidelines']
        )

        answer = agent.run(prompt)

        task_answer = {
            "task_id": str(tid),
            "agent_answer": str(answer),
            "reasoning_trace": str(clean_reasoning_trace(agent.logs))
        }

        agent_answers.append(task_answer)

    return agent_answers
