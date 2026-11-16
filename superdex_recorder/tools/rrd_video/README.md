# rrd_video

Renders superdex_recorder `.rrd` recordings to videos at 1x real time. Rerun draws every
frame from the recording, headless; nothing is re-simulated or redrawn by hand.

## Setup

Python 3.11+ with the packages in `requirements.txt` (Rerun 0.33 brings the viewer, and
imageio-ffmpeg a static ffmpeg):

```sh
pip install -r superdex_recorder/tools/rrd_video/requirements.txt
```

The viewer uses the machine's graphics driver. On a machine without a GPU or display,
install Mesa's CPU Vulkan driver, lavapipe (e.g. the `mesa-vulkan-drivers` package), or
point `RRD_VIDEO_LAVAPIPE` at a lavapipe install that is not on the system path (a
directory holding `lib/**/libvulkan_lvp.so` and `share/vulkan/icd.d/lvp_icd*.json`). It is
tested on Linux.

## Render

```sh
cd superdex_recorder/tools
python -m rrd_video render episode.rrd [more.rrd ...] --out videos/
```

Each recording writes:

| File | Content |
|---|---|
| `<stem>.webm` | VP9, yuv420p, BT.709 limited range: plays in browsers |
| `<stem>.mp4` | H.264, with `--mp4` |
| `<stem>.video.json` | Manifest (`superdex.rrd_video/1`): input SHA-256, recorded clock, fps, camera and where it came from, in-view check, tool versions |
| `<stem>.first.png`, `<stem>.last.png` | First and last video frames |

Without `--out`, outputs go beside each input. Every input is checked before rendering
starts, and outputs are written only after every input has rendered: local files all at
once, remote ones in order with manifests last, so a video without its `.video.json` is
incomplete. Errors exit 1 with a one-line reason (usage errors exit 2). Refused inputs: recordings that
are not complete superdex_recorder episodes (no `/episode` schema, no footer), recordings
with fewer than two frames, and recordings with `Clear` events or named transform frames.
A frame where the view shows only background is an error: the camera misses the scene.

Options: `--fps` (default 60), `--start`/`--end` (seconds after the first recorded
frame), `--size 1280x720`, `--exclude '/world/cameras/**'` (leave entities out of the view
and of the automatic camera), `--mp4`, `--workers`.

## Timing

Video frame i shows the recorded frame nearest to sim_time t0 + i / fps, the earlier one on
a tie, so the video plays at 1x at any rate from 1 to 240 fps. Below the recorded rate
frames are skipped; above it they repeat evenly, and each distinct frame is rendered once.
`--fps native` keeps the recorded rate: every frame of a constant-step recording, every
k-th one above 240 Hz, and for a recording without a constant step the largest of 60, 50,
30, 25, 20 or 10 fps at or below its recorded rate. A recorded rate within 0.1% of a whole
number plays at that number (59.9988 Hz at 60 fps); an `--fps` value is used as given.

## Camera

One fixed camera per video, from the first of:

1. `--camera 'eye=x,y,z target=x,y,z [up=x,y,z]'` or a camera JSON file. `--camera auto`
   always picks automatically.
2. The camera saved for the recording's task in `--registry DIR` (one JSON file per task).
   Nothing is saved or read unless a workflow passes a registry.
3. An automatic pick. It finds what the task is about from the recording alone: bodies that
   move and are not the robot, the robot links while they are near them, and the fixtures
   they touch (`--framing wide` adds every moving robot link). It chooses the view that sees
   them unoccluded, with their motion across the image, over the recorded frames inside
   `--start`/`--end`, whatever the `--fps`.

Recordings of one task rendered together share the pick of the first of them by path,
unless it keeps less than 90% of a recording's subject in view; so the other recordings in
a run can change a recording's camera. The task key comes from the episode properties
`scene.task.filename`, `task`, `env` or `scene`; recordings without them never share.

`camera show` prints the task key and the camera a render would use. `camera pick --out
DIR` writes stills of the four best candidates and `<stem>.camera.json`: the best camera,
usable as `--camera`, with every candidate's camera and scores. A workflow that
wants to reuse a camera saves it in a directory of its choice and passes the same
`--registry` to later renders:

```sh
python -m rrd_video camera save --registry cams/ --recording episode.rrd --from episode.camera.json
python -m rrd_video camera save --registry cams/ --task task=my_task --from 'eye=1,1,1 target=0,0,0'
python -m rrd_video render --registry cams/ episode.rrd --out videos/
```

The up axis comes from the recorded gravity, else from flat static bodies, else +Z (with a
warning); if the video looks rolled, pass the right axis with `--up` (e.g. `--up y`).

## Environment

| Variable | Use |
|---|---|
| `RRD_VIDEO_LAVAPIPE` | A lavapipe install for CPU rendering (see Setup) |
| `RRD_VIDEO_RERUN_CLI` | The Rerun CLI to use instead of the rerun-sdk package's |
| `RRD_VIDEO_FFMPEG` | The ffmpeg to use instead of imageio-ffmpeg's (needs libvpx-vp9) |
| `RRD_VIDEO_MEMORY_BYTES` | The memory budget for parallel viewers, if not the cgroup's |

## Tests

```sh
cd superdex_recorder/tools
python -m unittest discover -s rrd_video/tests -t .
```

The render tests need a working viewer driver (see Setup).
