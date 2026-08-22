# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-strict

import glob
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from lerobot.datasets.lerobot_dataset import (
    HF_LEROBOT_HOME,
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from torch.utils.data import DataLoader


def get_splits(repo_id: str) -> list[str]:
    """
    Get the list of splits for a given repo_id.
    """
    splits = []
    split_dirs = map(Path, glob.glob(f"{HF_LEROBOT_HOME / repo_id}/*"))
    for split_dir in split_dirs:
        # add to result if valid
        if split_dir.is_dir() and (split_dir / "meta/info.json").exists():
            splits.append(split_dir.name)
    return splits


def load_dataset(repo_id: str) -> dict[str, LeRobotDataset]:
    """
    Loads a dataset and all its splits given a repo_id.
    """
    datasets = {}
    for split in get_splits(repo_id):
        datasets[split] = LeRobotDataset(f"{repo_id}/{split}")
    return datasets


def make_dataloader(
    repo_id: str, batch_size: int, shuffle: bool = True
) -> tuple[dict[str, DataLoader], dict[str, LeRobotDatasetMetadata]]:
    """
    Returns a dataloader and a LeRobotDatasetMetadata for each split associated in a given repo_id.
    """
    datasets = load_dataset(repo_id)
    dataloaders = {}
    dataloaders_meta = {}
    for split, dataset in datasets.items():
        dataloaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        dataloaders_meta[split] = dataset.meta
    return dataloaders, dataloaders_meta


def load_trajectories(repo_id: str) -> List[List[Dict[str, Any]]]:
    """Load a list of action trajectories from a recorded dataset."""
    if str(repo_id).endswith(".h5"):
        return parse_h5(repo_id)
    datasets = load_dataset(repo_id=repo_id)
    trajectories = []
    for _, dataset in datasets.items():
        l_start = dataset.episode_data_index["from"].tolist()
        l_end = dataset.episode_data_index["to"].tolist()
        for start, end in zip(l_start, l_end):
            trajectories.append([dataset[x] for x in range(start, end)])
    return trajectories


def parse_h5(h5_file: str) -> dict[str, np.ndarray]:
    import h5py

    handle = {
        "robot": "76",
        "box": "2",
        "lid": "4",
        "shape": "6",
    }
    trajectories = []

    with h5py.File(h5_file, "r") as h5:
        for data_t in h5["events"].values():
            if data_t.attrs["eventType"].decode() == "Step":
                assert "actors" in data_t.keys()
                assert handle["robot"] in data_t["actors"].keys()

                actor_data = data_t["actors"]
                actor_names = {
                    k: data_t["actors"][k].attrs["name"] for k in actor_data.keys()
                }

                positions = {
                    k: actor_data[k].attrs["translation"].round(5)
                    for k in actor_data.keys()
                    if "translation" in actor_data[k].attrs.keys()
                }
                orientations = {
                    k: actor_data[k].attrs["rotation"].round(5)
                    for k in actor_data.keys()
                    if "rotation" in actor_data[k].attrs.keys()
                }

                if len(trajectories) == 0:
                    for k, v in positions.items():
                        pos = [v[0], -v[1], -v[2]]
                        ori = orientations[k]
                        print(k, actor_names[k], pos, ori)
                    print("#" * 10)

                # add step
                trajectories.append(
                    {
                        "object_pos": positions[handle["shape"]],
                        "action": np.array(
                            list(data_t["actors"][handle["robot"]]["targetPose"])
                        ),
                    }
                )
    return [trajectories]
