from dotenv import load_dotenv
from smolagents import CodeAgent
from smolagents.models import OpenAIServerModel

from utils import download_dataset, CONTEXT_FILENAMES, setup_langfuse

load_dotenv()

setup_langfuse()
context_files = download_dataset()


MODEL_ID = "gpt-4o-mini-2024-07-18" # "o3-mini-2025-01-31"
MAX_STEPS = 7

agent = CodeAgent(
    tools=[],
    model=OpenAIServerModel(MODEL_ID),
    additional_authorized_imports=["numpy", "pandas", "json", "csv", "os", "glob", "markdown"],
    max_steps=MAX_STEPS,
)
# give agent power to open files
agent.python_executor.send_tools({"open": open})

PROMPT = """You are an expert data analyst and you will answer factoid questions by loading and referencing the files/documents listed below.
You have these files available:
{context_files}
Don't forget to reference any documentation in the data dir before answering a question.

Here is the question you need to answer:
{question}

Here are the guidelines you must follow when answering the question above:
{guidelines}
"""
question = "What are the unique set of merchants in the payments data?"
guidelines = "Answer must be a comma separated list of the merchant names. If a question does not have a relevant or applicable answer for the task, please respond with 'Not Applicable'."

PROMPT = PROMPT.format(
    context_files=context_files,
    question=question,
    guidelines=guidelines
)

answer = agent.run(PROMPT)
