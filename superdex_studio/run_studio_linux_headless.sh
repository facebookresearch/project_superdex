#!/bin/bash
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

# Run SuperDex Studio on headless Linux via Xvfb
# Usage:
#   ./run_studio_linux_headless.sh              # Interactive (Ctrl-C to stop)
#   ./run_studio_linux_headless.sh --background # Background mode (prints PID)
#   ./run_studio_linux_headless.sh --kill       # Kill background instance
#   ./run_studio_linux_headless.sh --display 42 # Use display :42 instead of :99
set -euo pipefail

WIDTH=1920
HEIGHT=1080
DISPLAY_NUM=99
BACKGROUND=false
PIDFILE="/tmp/superdex_studio.pid"
XVFB_PIDFILE="/tmp/superdex_studio_xvfb.pid"
MODE="@arvr/mode/linux/opt"

while [[ $# -gt 0 ]]; do
    case $1 in
        --display)
            [[ -n "${2:-}" ]] || { echo "Error: --display requires a value"; exit 1; }
            DISPLAY_NUM="$2"; shift 2;;
        --background) BACKGROUND=true; shift;;
        --kill)
            if [ -f "$PIDFILE" ]; then
                PID=$(cat "$PIDFILE")
                if kill -0 "$PID" 2>/dev/null && grep -q "superdex_studio\|buck2" /proc/"$PID"/cmdline 2>/dev/null; then
                    kill "$PID" && echo "Killed app" || echo "App not running"
                else
                    echo "App not running (stale PID file)"
                fi
                rm -f "$PIDFILE"
            else
                echo "No app PID file found"
            fi
            if [ -f "$XVFB_PIDFILE" ]; then
                kill "$(cat "$XVFB_PIDFILE")" 2>/dev/null && echo "Killed Xvfb" || echo "Xvfb not running"
                rm -f "$XVFB_PIDFILE"
            fi
            exit 0;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

# Ensure Xvfb is installed
if ! command -v Xvfb &>/dev/null; then
    echo "Error: Xvfb is not installed."
    echo "  Fedora/RHEL: sudo dnf install xorg-x11-server-Xvfb"
    echo "  Ubuntu/Debian: sudo apt-get install xvfb"
    exit 1
fi

# Start Xvfb if needed
STARTED_XVFB=false
if ! xdpyinfo -display ":$DISPLAY_NUM" &>/dev/null; then
    Xvfb ":$DISPLAY_NUM" -screen 0 "${WIDTH}x${HEIGHT}x24" &
    XVFB_PID=$!
    STARTED_XVFB=true

    # Wait for Xvfb to be ready (up to 5 seconds)
    for _ in $(seq 1 50); do
        if xdpyinfo -display ":$DISPLAY_NUM" &>/dev/null; then
            break
        fi
        sleep 0.1
    done

    if ! xdpyinfo -display ":$DISPLAY_NUM" &>/dev/null; then
        echo "[MCP] Error: Xvfb failed to start on display :$DISPLAY_NUM"
        kill "$XVFB_PID" 2>/dev/null
        exit 1
    fi
fi

export DISPLAY=":$DISPLAY_NUM"

echo "[MCP] Building SuperDex Studio..."
buck2 build //arvr/projects/superdex/superdex_studio:superdex_studio "$MODE"

if [ "$BACKGROUND" = true ]; then
    # Note: captures buck2 wrapper PID. Signal propagation to the child app depends on buck2 behavior.
    buck2 run //arvr/projects/superdex/superdex_studio:superdex_studio "$MODE" &
    APP_PID=$!
    echo "$APP_PID" > "$PIDFILE"
    if [ "$STARTED_XVFB" = true ]; then
        echo "$XVFB_PID" > "$XVFB_PIDFILE"
    fi
    echo "[MCP] Running in background (PID: $APP_PID)"
    echo "[MCP] Kill with: $0 --kill"
else
    if [ "$STARTED_XVFB" = true ]; then
        trap 'kill $XVFB_PID 2>/dev/null' EXIT
    fi
    echo "[MCP] Press Ctrl-C to stop"
    buck2 run //arvr/projects/superdex/superdex_studio:superdex_studio "$MODE"
fi
