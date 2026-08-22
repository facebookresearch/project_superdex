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

import pathlib
import threading

import imageio
import numpy as np
import numpy.typing as npt

########################################################################################


class AnimationWriter:
    """
    Utility class allowing to write animations to disk. The animations are written in
    separate threads to avoid blocking the main thread.
    """

    # NOTE: This is perhaps the simplest (and possibly dirtiest) parallel animation
    # writer ever, but it works. I tried to use the multiprocessing module, but it
    # ended up producing hiccups in the main thread. Revisit this later if it becomes
    # a problem.

    ####################################################################################
    # Members
    ####################################################################################

    # Private members.
    _output_path: pathlib.Path
    _images: list[npt.NDArray]
    _fps: int
    _fmt: str
    _jobs: list[threading.Thread]

    ####################################################################################
    # Constructor
    ####################################################################################

    def __init__(self, output_path: pathlib.Path, fps: int, fmt: str) -> None:
        """
        Initializes the animation writer. The output path is the directory where the
        animations will be written to. If the directory does not exist, it will be
        created.
        """

        self._output_path = output_path
        self._output_path.mkdir(parents=True, exist_ok=True)
        self._images = []
        self._fps = fps
        self._fmt = fmt
        self._jobs = []

    def __del__(self) -> None:
        """Flush the animations to disk before the object is destroyed."""
        self.flush()

    ####################################################################################
    # Methods
    ####################################################################################

    def add(self, image: npt.NDArray | list[npt.NDArray]) -> None:
        """
        Adds the given frame or sequence to the animation. If a frame is given, it must
        be a numpy array of shape (H, W, D). If a sequence is given, it must be either a
        numpy array of shape (N, H, W, D) or a list of numpy arrays of shape (H, W, D).
        Note that the depth must be 3 or 4, where 3 means RGB and 4 means RGBA. The
        alpha channel is always ignored.
        """

        # Convert the image or sequence to a numpy array.
        # Check if the number of dimensions is valid.
        image = np.asarray(image)
        if image.ndim not in (3, 4) or not 3 <= image.shape[-1] <= 4:
            raise ValueError("Expected a RGB(A) image or sequence.")

        # Drop the alpha channel if present.
        if image.shape[-1] == 4:
            image = image[..., 0:3]

        # If this is not the first frame, check if dimensions agree.
        if len(self._images) > 0:
            shape = image.shape
            expected_shape = self._images[0].shape
            if shape != expected_shape:
                raise ValueError(
                    "Dimensions of the given image(s) do not match: "
                    f"{shape} vs {expected_shape}"
                )

        # Imageio expects a sequence of images rather than a single tensor with all the
        # frames. Therefore, we split the given tensor into a list of images.
        if image.ndim == 3:
            image = [image]  # Wrap in list to uniformly use extend.
        self._images.extend(image)

    def write(self, name: str) -> None:
        """
        Writes the animation to a file with the given name. The name is the name of the
        file without the extension. The extension is determined by the format specified
        in the constructor.
        """

        # Prepare arguments to the writer thread.
        fpath = self._output_path / f"{name}.{self._fmt}"
        args = {"fpath": fpath, "images": list(self._images), "fps": self._fps}
        self._images = []

        # Spawn the writer thread, remove finished jobs, and run the new thread.
        thread = threading.Thread(target=self._write_image_to_disk, kwargs=args)
        self._remove_finished_jobs()
        self._jobs.append(thread)
        thread.start()

    def flush(self) -> None:
        """
        Flushes the animations to disk. This function blocks until all animations have
        been written to disk.
        """
        for thread in self._jobs:
            if thread.is_alive():
                thread.join()
        self._jobs = []

    def has_frames(self) -> bool:
        """Returns true if the animation writer has any frames to write."""
        return len(self._images) > 0

    def _remove_finished_jobs(self) -> None:
        self._jobs = [thread for thread in self._jobs if thread.is_alive()]

    @staticmethod
    def _write_image_to_disk(
        fpath: pathlib.Path, images: list[npt.NDArray], fps: int
    ) -> None:
        imageio.mimsave(uri=str(fpath), ims=images, fps=fps)
