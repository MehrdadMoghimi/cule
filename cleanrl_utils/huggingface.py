# Adapted from CleanRL's cleanrl_utils/huggingface.py
# (https://github.com/vwxyzjn/cleanrl, MIT; license in cleanrl/LICENSE.md).
#
# Differences from upstream: the model card credits this fork (the trainers
# here are modified and several algorithms do not exist upstream, so pointing
# users at `pip install cleanrl` would be wrong), the poetry.lock/pyproject
# upload is dropped because this repository does not use poetry, and the
# `tenacity` dependency is replaced with a small retry loop.
"""Optional Hugging Face Hub upload for the bundled trainers (`--upload-model`)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from pprint import pformat
from typing import Sequence

import numpy as np

HUGGINGFACE_VIDEO_PREVIEW_FILE_NAME = "replay.mp4"
HUGGINGFACE_README_FILE_NAME = "README.md"

REPO_URL = "https://github.com/MehrdadMoghimi/cule"


def model_card(
    args: argparse.Namespace,
    repo_id: str,
    algo_name: str,
    script_name: str,
    command: str,
) -> str:
    """Build the model card. Kept separate so it can be rendered without network."""
    return f"""
# **{algo_name}** Agent Playing **{args.env_id}**

This is a trained model of a {algo_name} agent playing {args.env_id}.

It was trained with the CuLE-accelerated CleanRL-style trainers in
[{REPO_URL}]({REPO_URL}), a fork of [NVlabs/cule](https://github.com/NVlabs/cule).
The trainers are adapted from [CleanRL](https://github.com/vwxyzjn/cleanrl) and
[LeanRL](https://github.com/meta-pytorch/LeanRL); see the header of
`{script_name}` for the provenance of this particular algorithm.

## Get started

```bash
git clone {REPO_URL}
cd cule
pip install -r requirements.txt
CUDA_HOME=/usr/local/cuda pip install --no-build-isolation -e .
```

## Command to reproduce the training

```bash
python cleanrl/{script_name} {command}
```

# Hyperparameters
```python
{pformat(vars(args))}
```
"""


def push_to_hub(
    args: argparse.Namespace,
    episodic_returns: Sequence[float],
    repo_id: str,
    algo_name: str,
    folder_path: str,
    video_folder_path: str = "",
    revision: str = "main",
    create_pr: bool = False,
    private: bool = False,
    attempts: int = 5,
) -> str:
    # Imported lazily so that `--upload-model` is the only thing that needs
    # huggingface_hub installed.
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
    from huggingface_hub.repocard import metadata_eval_result, metadata_save

    api = HfApi()
    repo_url = api.create_repo(repo_id=repo_id, exist_ok=True, private=private)
    entity, repo = repo_url.split("/")[-2:]
    repo_id = f"{entity}/{repo}"

    # Remove the previous run's event files and videos.
    operations: list = [
        CommitOperationDelete(path_in_repo=file)
        for file in api.list_repo_files(repo_id=repo_id)
        if ".tfevents" in file or file.endswith(".mp4")
    ]

    script_path = Path(sys.argv[0])
    readme_path = Path(folder_path) / HUGGINGFACE_README_FILE_NAME
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(
        model_card(args, repo_id, algo_name, script_path.name, " ".join(sys.argv[1:])),
        encoding="utf-8",
    )

    metadata = {
        "tags": [
            args.env_id,
            "deep-reinforcement-learning",
            "reinforcement-learning",
            "custom-implementation",
            "cule",
        ]
    }
    metadata.update(
        metadata_eval_result(
            model_pretty_name=algo_name,
            task_pretty_name="reinforcement-learning",
            task_id="reinforcement-learning",
            metrics_pretty_name="mean_reward",
            metrics_id="mean_reward",
            metrics_value=f"{np.average(episodic_returns):.2f} +/- {np.std(episodic_returns):.2f}",
            dataset_pretty_name=args.env_id,
            dataset_id=args.env_id,
        )
    )
    metadata_save(readme_path, metadata)

    if video_folder_path:
        video_files = list(Path(video_folder_path).glob("*.mp4"))
        operations += [
            CommitOperationAdd(path_or_fileobj=str(file), path_in_repo=str(file)) for file in video_files
        ]
        if video_files:
            latest_file = max(video_files, key=lambda file: int("".join(filter(str.isdigit, file.stem)) or 0))
            operations.append(
                CommitOperationAdd(
                    path_or_fileobj=str(latest_file), path_in_repo=HUGGINGFACE_VIDEO_PREVIEW_FILE_NAME
                )
            )

    operations += [
        CommitOperationAdd(path_or_fileobj=str(item), path_in_repo=str(item.relative_to(folder_path)))
        for item in Path(folder_path).glob("*")
        if item.is_file()
    ]
    # Upload the training script itself so the run stays reproducible.
    if script_path.is_file():
        operations.append(CommitOperationAdd(path_or_fileobj=str(script_path), path_in_repo=script_path.name))

    for attempt in range(1, attempts + 1):
        try:
            api.create_commit(
                repo_id=repo_id,
                operations=operations,
                commit_message="pushing model",
                revision=revision,
                create_pr=create_pr,
            )
            break
        except Exception as error:  # network/rate-limit flakiness
            if attempt == attempts:
                raise
            print(f"push attempt {attempt}/{attempts} failed ({error}); retrying")
            time.sleep(3)

    print(f"Model pushed to {repo_url}")
    return repo_url
