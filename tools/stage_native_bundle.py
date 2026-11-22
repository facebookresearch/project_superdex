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

"""Stage merged ISA images with import modules that share one native library."""

import argparse
import filecmp
import json
import shutil
import stat
import subprocess
from pathlib import Path


def dependencies(path: Path, objdump: str) -> list[str]:
    result = subprocess.run(
        [objdump, "-p", str(path)], check=True, capture_output=True, text=True
    )
    return [
        parts[1]
        for line in result.stdout.splitlines()
        if len(parts := line.split()) == 2 and parts[0] == "NEEDED"
    ]


def stage(  # noqa: C901
    spec: dict, output: Path, objdump: str, patchelf: str, strip: str | None
) -> None:
    manifest = {
        "precision": spec["precision"],
        "variants": {},
        "images": {},
    }
    baseline_modules = set(spec["avx2"]["modules"])
    # Runtime shims import namespaced modules by one name for every ISA.
    avx512_modules = {
        name if "." in name else name.removesuffix("_avx512")
        for name in spec["avx512"]["modules"]
    }
    if not baseline_modules or baseline_modules != avx512_modules:
        raise ValueError("Merged ISA images must expose matching native modules")
    for isa in ("avx2", "avx512"):
        inventory = spec[isa]
        image = Path(inventory["image"])
        image_name = image.name
        image_needed = dependencies(image, objdump)
        separate_native = [
            name
            for name in image_needed
            if name.startswith(("libmochi", "libsuperdex"))
        ]
        if separate_native:
            raise ValueError(
                f"Merged {isa} image still depends on separate native libraries: {separate_native}"
            )
        candidates = {}
        for library in inventory["libraries"]:
            soname = library["soname"]
            if soname is None:
                soname = Path(library["soname_file"]).read_text().strip()
            candidates.setdefault(soname, []).append(Path(library["source"]))
        pending = [(image_name, image)]
        entries = {}
        sources = {}
        destination = output / isa
        destination.mkdir(parents=True)
        while pending:
            name, source = pending.pop()
            if name in entries:
                if sources[name] != source and not filecmp.cmp(
                    sources[name], source, shallow=False
                ):
                    raise ValueError(f"Conflicting {isa} native files named {name}.")
                continue
            if ("_avx512" in name) != (isa == "avx512") and name.startswith(
                ("libmochi", "libsuperdex", "mochi_", "superdex_")
            ):
                raise ValueError(f"Mixed native ISA: {isa}/{name}")
            with source.open("rb") as native_file:
                header = native_file.read(20)
            if (
                len(header) != 20
                or header[:6] != b"\x7fELF\x02\x01"
                or int.from_bytes(header[18:20], "little") != 62
            ):
                raise ValueError(
                    f"Native bundles require Linux/x64 ELF files: {source}."
                )
            if name.startswith(("libmochi", "libsuperdex", "mochi_", "superdex_")) and (
                "_double" in name
            ) != (spec["precision"] == "double"):
                raise ValueError(f"Mixed native precision: {spec['precision']}/{name}")
            needed = dependencies(source, objdump)
            installed = destination / name
            shutil.copy2(source, installed)
            installed.chmod(installed.stat().st_mode | stat.S_IWUSR)
            subprocess.run(
                [patchelf, "--set-rpath", "$ORIGIN", str(installed)], check=True
            )
            if strip is not None:
                subprocess.run([strip, "-g", str(installed)], check=True)
            sources[name] = source
            entries[name] = {
                "path": f"{isa}/{name}",
                "needed": needed,
            }
            for dependency in needed:
                if (
                    dependency.startswith(("libmochi", "libsuperdex"))
                    and dependency not in candidates
                ):
                    raise ValueError(
                        f"Missing {isa} native dependency {dependency} required by {name}."
                    )
                for candidate in candidates.get(dependency, []):
                    pending.append((dependency, candidate))
        for module in inventory["modules"]:
            if not all(part.isidentifier() for part in module.split(".")):
                raise ValueError(f"Invalid {isa} native module name: {module}")
            if "." in module:
                continue
            if ("_avx512" in module) != (isa == "avx512"):
                raise ValueError(f"Invalid {isa} native module alias: {module}")
            if ("_double" in module) != (spec["precision"] == "double"):
                raise ValueError(
                    f"Mixed native precision: {spec['precision']}/{module}"
                )
            alias = module + ".py"
            # Resource packaging can dereference symlinks. Load every module from the
            # same image path so the dynamic loader keeps one native dependency state.
            # An explicit loader keeps the image loadable whatever the platform's
            # extension suffixes are.
            (destination / alias).write_text(
                "import importlib.machinery\n"
                "import importlib.util\n"
                "import sys\n"
                "from pathlib import Path\n\n"
                f"_image = str(Path(__file__).with_name({image_name!r}))\n"
                "_loader = importlib.machinery.ExtensionFileLoader(__name__, _image)\n"
                "_spec = importlib.util.spec_from_file_location(\n"
                "    __name__, _image, loader=_loader\n"
                ")\n"
                "_module = importlib.util.module_from_spec(_spec)\n"
                "_spec.loader.exec_module(_module)\n"
                "sys.modules[__name__] = _module\n"
            )
            entries[alias] = {
                "path": f"{isa}/{alias}",
                "needed": image_needed,
                "image": f"{isa}/{image_name}",
            }
        manifest["variants"][isa] = entries
        manifest["images"][isa] = f"{isa}/{image_name}"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--objdump", required=True)
    parser.add_argument("--patchelf", required=True)
    parser.add_argument("--strip")
    args = parser.parse_args()
    stage(
        json.loads(args.spec.read_text()),
        args.out,
        args.objdump,
        args.patchelf,
        args.strip,
    )


if __name__ == "__main__":
    main()
