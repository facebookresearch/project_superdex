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

"""Frame banks: recorded episodes in one file, ready to restore any frame of.

A bank file (``.superdex_framebank``) holds superdex_recorder's ``FrameBank`` in Mochi
reflection's binary format. It is tied to the bank structs and the Mochi precision of the
build that packed it, so re-pack it from its recordings after either changes. The SuperDex
Lab reads fp32 banks.

Pack recordings and describe the result from the command line::

    python -m superdex.recorder.frame_bank pack --out bank.superdex_framebank a.rrd b.rrd
    python -m superdex.recorder.frame_bank info bank.superdex_framebank
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from . import _extension


def pack(
    recordings: Sequence[str | os.PathLike[str]], out: str | os.PathLike[str]
) -> None:
    """Pack recorded episodes into a bank file, in the order given.

    All episodes must record the same actors. Raises ``superdex.physics.Error`` naming
    the recording that cannot be packed.
    """
    _extension._pack_frame_bank([Path(path) for path in recordings], Path(out))


def info(path: str | os.PathLike[str]) -> str:
    """A summary of a bank file: its episodes, frames and precision, its actors (and
    whether a static actor's pose varies across episodes) and its controllers' target
    fields."""
    return _extension._describe_frame_bank(Path(path))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m superdex.recorder.frame_bank",
        description="Pack recorded episodes into a frame bank file, or describe one.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    pack_command = commands.add_parser(
        "pack", help="Pack recordings into a bank, ordered by their full paths."
    )
    pack_command.add_argument("--out", required=True, type=Path, help="The bank file.")
    pack_command.add_argument(
        "recordings", nargs="+", type=Path, help="Episodes (.rrd)."
    )
    info_command = commands.add_parser("info", help="Describe a bank file.")
    info_command.add_argument("bank", type=Path, help="The bank file.")
    args = parser.parse_args(argv)
    if args.command == "pack":
        pack(sorted(os.path.abspath(path) for path in args.recordings), args.out)
        print(info(args.out))
    else:
        print(info(args.bank))


if __name__ == "__main__":
    main()
