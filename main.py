from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

ds = load_dataset("adyen/DABstep", name="tasks", split="default")
assert len(ds) == 450, "Number of tasks is not correct"

for task in ds:
    print(task)
    break
    # agent solves task

# write tasks answers to file in the format provided in the leaderboard
# submit file to the form in the leaderboard
