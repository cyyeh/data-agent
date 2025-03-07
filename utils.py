import os
import shutil
from huggingface_hub import hf_hub_download


def download_dataset(data_destination_dir: str = "/tmp/DABstep-data"):
    if os.path.exists(data_destination_dir):
        shutil.rmtree(data_destination_dir)

    CONTEXT_FILENAMES = [
        "data/context/acquirer_countries.csv",
        "data/context/payments-readme.md",
        "data/context/payments.csv",
        "data/context/merchant_category_codes.csv",
        "data/context/fees.json",
        "data/context/merchant_data.json",
        "data/context/manual.md",
    ]

    for filename in CONTEXT_FILENAMES:
        hf_hub_download(
            repo_id="adyen/DABstep",
            repo_type="dataset",
            filename=filename,
            local_dir=data_destination_dir,
            force_download=True
        )

    CONTEXT_FILENAMES = [f"{data_destination_dir}/{filename}" for filename in CONTEXT_FILENAMES]

    for file in CONTEXT_FILENAMES:
        assert os.path.exists(file), f"{file} does not exist."
