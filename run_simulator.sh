#!/bin/bash
# Positional args: $1 $2 $3 $4 as before ($4 = wide FOV for quasar). Any further args ($5, $6, …) are appended to the simulator command line.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXTRA_APP_ARGS=()
DUMP_OUTPUT=0
for arg in "${@:5}"; do
    if [ "$arg" == "dump" ]; then
        DUMP_OUTPUT=1
    else
        EXTRA_APP_ARGS+=("$arg")
    fi
done
WIDE_FOV_USAGE_ROOT="/media/wuhaolu/c54fff3f-cab5-4dcf-94c3-c83855e5a9bd/quasarOutput/test"

has_extra_arg() {
    local needle="$1"
    for arg in "${EXTRA_APP_ARGS[@]}"; do
        if [ "$arg" == "$needle" ] || [[ "$arg" == "$needle="* ]]; then
            return 0
        fi
    done
    return 1
}

uses_recorded_texel_usage_mask() {
    local prev_arg=""
    for arg in "${EXTRA_APP_ARGS[@]}"; do
        if [ "$prev_arg" == "--wide-fov-mask-method" ]; then
            case "$arg" in
                recorded-texel-usage|recorded_texel_usage|texel-usage|mask-gt)
                    return 0
                    ;;
            esac
        fi
        case "$arg" in
            --wide-fov-mask-method=recorded-texel-usage|\
            --wide-fov-mask-method=recorded_texel_usage|\
            --wide-fov-mask-method=texel-usage|\
            --wide-fov-mask-method=mask-gt)
                return 0
                ;;
        esac
        prev_arg="$arg"
    done
    return 1
}

uses_stencil_gt_pose_mask() {
    local prev_arg=""
    for arg in "${EXTRA_APP_ARGS[@]}"; do
        if [ "$prev_arg" == "--wide-fov-mask-method" ]; then
            case "$arg" in
                stencilwithgtpose|stencil-with-gt-pose|stencil_gt_pose)
                    return 0
                    ;;
            esac
        fi
        case "$arg" in
            --wide-fov-mask-method=stencilwithgtpose|\
            --wide-fov-mask-method=stencil-with-gt-pose|\
            --wide-fov-mask-method=stencil_gt_pose)
                return 0
                ;;
        esac
        prev_arg="$arg"
    done
    return 1
}

generate_wide_fov_ground_truth_pose_records() {
    local output_dir="$1"
    local logged_info_dir="$output_dir/loggedInfo"
    local build_log="$logged_info_dir/build_log.txt"
    local legacy_build_log="$output_dir/build_log.txt"
    local generator_script="$logged_info_dir/generate_wide_fov_ground_truth_pose_records.py"
    local records_json="$logged_info_dir/prev_curr_pose_records.json"
    local python_bin=""

    if [ ! -f "$build_log" ]; then
        if [ -f "$legacy_build_log" ]; then
            build_log="$legacy_build_log"
        else
            return
        fi
    fi

    mkdir -p "$logged_info_dir"

    if command -v python3 >/dev/null 2>&1; then
        python_bin="python3"
    elif command -v python >/dev/null 2>&1; then
        python_bin="python"
    else
        echo "Python not found; skipping remote-to-recorded frame map generation for $build_log"
        return
    fi

    cat > "$generator_script" <<'PY'
#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path

build_log = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("build_log.txt")
output_json = (
    Path(sys.argv[2])
    if len(sys.argv) > 2
    else Path(__file__).with_name("prev_curr_pose_records.json")
)

remote_frame_re = re.compile(r"\bFrame ID:\s*(\d+)\b")
recorded_frame_re = re.compile(r"Do recording for frame:\s*(\d+)\b")
remote_pose_re = re.compile(r"Remote Render with pose: pos=\(([^)]*)\), rot=\(([^)]*)\)")
recorded_pose_re = re.compile(r"\bRender with pose: pos=\(([^)]*)\), rot=\(([^)]*)\)")

pose_columns = [
    "tx",
    "ty",
    "tz",
    "rx_deg",
    "ry_deg",
    "rz_deg",
]
remote_to_recorded = {}
current_remote_frame = None
pending_remote_pose = None
latest_recorded_pose = None

def parse_pose(match):
    values = []
    for group_index in (1, 2):
        values.extend(part.strip() for part in match.group(group_index).split(","))
    if len(values) != len(pose_columns):
        return None
    return values

def get_remote_frame(remote_frame_id):
    return remote_to_recorded.setdefault(
        remote_frame_id,
        {
            "remote_pose": None,
            "recorded_frames": [],
        },
    )

def pose_values(pose):
    return pose if pose is not None else [0.0] * len(pose_columns)

def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

def make_pose_json(pose):
    tx, ty, tz, rx, ry, rz = [as_float(value) for value in pose_values(pose)]
    return {
        "position": {
            "x": tx,
            "y": ty,
            "z": tz,
        },
        "euler_rotation_degrees": {
            "x": rx,
            "y": ry,
            "z": rz,
        },
    }

with build_log.open("r", encoding="utf-8", errors="replace") as log_file:
    for line in log_file:
        remote_pose_match = remote_pose_re.search(line)
        if remote_pose_match:
            pending_remote_pose = parse_pose(remote_pose_match)
            continue

        if "Remote Render with pose" not in line:
            recorded_pose_match = recorded_pose_re.search(line)
            if recorded_pose_match:
                latest_recorded_pose = parse_pose(recorded_pose_match)
                continue

        remote_match = remote_frame_re.search(line)
        if remote_match:
            current_remote_frame = int(remote_match.group(1))
            get_remote_frame(current_remote_frame)["remote_pose"] = pending_remote_pose
            continue

        recorded_match = recorded_frame_re.search(line)
        if recorded_match and current_remote_frame is not None:
            recorded_frame_id = int(recorded_match.group(1))
            recorded_frame_id_png = max(recorded_frame_id - 1, 0)
            get_remote_frame(current_remote_frame)["recorded_frames"].append(
                (recorded_frame_id, recorded_frame_id_png, latest_recorded_pose)
            )

remote_render_frames = {}
recorded_frame_count = 0
for remote_frame_id in sorted(remote_to_recorded):
    remote_frame = remote_to_recorded[remote_frame_id]
    remote_pose_json = make_pose_json(remote_frame["remote_pose"])
    recorded_frames_json = {}
    for recorded_frame_id, recorded_frame_id_png, recorded_pose in remote_frame["recorded_frames"]:
        recorded_pose_json = make_pose_json(recorded_pose)
        recorded_pose_json["recorded_frame_id"] = recorded_frame_id
        recorded_pose_json["recorded_frame_id_png"] = recorded_frame_id_png
        recorded_frames_json[str(recorded_frame_id_png)] = recorded_pose_json
        recorded_frame_count += 1

    remote_render_frames[str(remote_frame_id)] = {
        "previous_pose": remote_pose_json,
        "current_pose": remote_pose_json,
        "delta_position": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
        },
        "position_shift_l2": 0.0,
        "Recorded Frame": recorded_frames_json,
    }

root = {
    "summary": {
        "source_build_log": str(build_log),
        "remote_frame_count": len(remote_render_frames),
        "remote_frames_with_recorded_count": sum(
            1 for frame in remote_render_frames.values() if frame["Recorded Frame"]
        ),
        "recorded_frame_count": recorded_frame_count,
        "recorded_frame_key": "recorded_frame_id_png",
    },
    "remote_render_frames": remote_render_frames,
}

output_json.parent.mkdir(parents=True, exist_ok=True)
with output_json.open("w", encoding="utf-8") as json_file:
    json.dump(root, json_file, indent=2, sort_keys=True)
    json_file.write("\n")

print(
    f"Wrote {output_json} with {len(remote_to_recorded)} remote frames "
    f"from {build_log}"
)
PY

    chmod +x "$generator_script"
    "$python_bin" "$generator_script" "$build_log" "$records_json"
}

organize_logged_info() {
    local output_dir="$1"
    local logged_info_dir="$output_dir/loggedInfo"

    if [ ! -d "$output_dir" ]; then
        return
    fi

    mkdir -p "$logged_info_dir"

    shopt -s nullglob
    local file
    for file in "$output_dir"/*.csv "$output_dir"/*.txt "$output_dir"/*.json "$output_dir"/*.py "$output_dir"/*.log; do
        if [ -f "$file" ]; then
            mv -f "$file" "$logged_info_dir/"
        fi
    done
    shopt -u nullglob
}

SCENES=(
    "viking_village" 
    "sponza" 
    # "robot_lab"
    "sun_temple"
    "san_miguel"
)

APP_REL=apps/scene_viewer/scene_viewer
PARAMS=()

# Add $4 as the widefov

if [ "$2" == "quasar" ]; then
    echo "Running Quasar build script..."
    APP_REL=apps/quasar/simulator/qr_simulator
    PARAMS+=("--pose-prediction" "--network-latency" "20.0" "--network-jitter" "0.0")
    PARAMS+=("--view-size" "0.25")
    PARAMS+=("--remote-fov-wide" "$4")
    if [[ "$3" =~ ^Confidence([0-9]+([.][0-9]+)?)$ ]]; then
        CONFIDENCE_SUFFIX="${BASH_REMATCH[1]}"
        if [[ "$CONFIDENCE_SUFFIX" == *.* ]]; then
            CONFIDENCE_VALUE="$CONFIDENCE_SUFFIX"
        elif [[ "$CONFIDENCE_SUFFIX" == "100" ]]; then
            CONFIDENCE_VALUE="1.0"
        else
            CONFIDENCE_VALUE="0.$CONFIDENCE_SUFFIX"
        fi
        echo "Using wide-FOV covariance confidence: $CONFIDENCE_VALUE"
        if ! has_extra_arg "--wide-fov-covariance-confidence"; then
            PARAMS+=("--wide-fov-covariance-confidence" "$CONFIDENCE_VALUE")
        fi
        if ! uses_stencil_gt_pose_mask && ! uses_recorded_texel_usage_mask; then
            if ! has_extra_arg "--wide-fov-covariance-samples"; then
                PARAMS+=("--wide-fov-covariance-samples" "5")
                echo "Using wide-FOV covariance samples: 5"
            fi
            if ! has_extra_arg "--wide-fov-covariance-debug-sampling"; then
                PARAMS+=("--wide-fov-covariance-debug-sampling")
                echo "Using wide-FOV covariance debug sampling"
            fi
        fi
    fi
    # PARAMS+=("--E-path")
elif [ "$2" == "baseline" ]; then
    echo "Running Baseline build script..."
    APP_REL=apps/scene_viewer/scene_viewer
    POSE_PREDICTOR=()
elif [ "$2" == "quadstream" ]; then
    echo "Running QuadStream build script..."
    APP_REL=apps/quadstream/simulator/qs_simulator
    PARAMS+=("--pose-prediction" "--pose-smoothing" "--network-latency" "20.0" "--network-jitter" "0.0")
elif [ "$2" == "meshwarp" ]; then
    echo "Running MeshWarp build script..."
    APP_REL=apps/meshwarp/simulator/mw_simulator
    PARAMS+=("--pose-prediction" "--pose-smoothing" "--network-latency" "20.0" "--network-jitter" "0.0")
elif [ "$2" == "atw" ]; then
    echo "Running ATW build script..."
    APP_REL=apps/atw/simulator/atw_simulator
    PARAMS+=("--pose-prediction" "--pose-smoothing" "--network-latency" "20.0" "--network-jitter" "0.0")
elif [ "$2" == "hybrid" ]; then
    echo "Running Hybrid build script..."
    APP_REL=apps/hybrid/simulator/hybrid_simulator
    PARAMS+=("--pose-prediction" "--pose-smoothing" "--network-latency" "20.0" "--network-jitter" "0.0")
    # PARAMS+=("--E-path")
else
    echo "Please provide a valid application name (e.g., quasar)."
    exit 1
fi

APP="$SCRIPT_DIR/$APP_REL"
if [ ! -x "$APP" ]; then
    BUILD_APP="$SCRIPT_DIR/build/$APP_REL"
    if [ -x "$BUILD_APP" ]; then
        APP="$BUILD_APP"
    fi
fi
APP_DIR="$(dirname "$APP")"
APP_BIN="./$(basename "$APP")"
APP_RUNNER=()
if [ "$DUMP_OUTPUT" -eq 1 ] && [ "${RUN_SIMULATOR_NO_XVFB:-0}" != "1" ]; then
    if [ -n "${DISPLAY:-}" ]; then
        echo "Using current DISPLAY=$DISPLAY for dump-mode GLFW/CUDA context creation"
    elif command -v xvfb-run >/dev/null 2>&1; then
        APP_RUNNER=("xvfb-run" "-a")
        echo "Using xvfb-run for dump-mode GLFW context creation"
    else
        echo "xvfb-run not found; dump-mode run will use the current DISPLAY"
    fi
fi

# Remove the directory if it exists
for SCENE in "${SCENES[@]}"; do
    DST_DIR=/media/wuhaolu/c54fff3f-cab5-4dcf-94c3-c83855e5a9bd/quasarOutput/$1/$SCENE/$2$3
    LOGGED_INFO_DIR="$DST_DIR/loggedInfo"
    BUILD_LOG="$LOGGED_INFO_DIR/build_log.txt"
    RUN_PARAMS=("${PARAMS[@]}")
    WIDE_FOV_USAGE_DIR="$WIDE_FOV_USAGE_ROOT/$SCENE/quasarWideFoVUsage"
    WIDE_FOV_LOGGED_INFO_DIR="$WIDE_FOV_USAGE_DIR/loggedInfo"

    if [ "$2" == "quasar" ]; then
        USE_RECORDED_TEXEL_USAGE=0
        if uses_recorded_texel_usage_mask; then
            USE_RECORDED_TEXEL_USAGE=1
        fi
        USE_STENCIL_GT_POSE=0
        if uses_stencil_gt_pose_mask; then
            USE_STENCIL_GT_POSE=1
        fi

        if [ "$USE_RECORDED_TEXEL_USAGE" -eq 1 ]; then
            generate_wide_fov_ground_truth_pose_records "$WIDE_FOV_USAGE_DIR"
            if ! has_extra_arg "--wide-fov-ground-truth-dir"; then
                RUN_PARAMS+=("--wide-fov-ground-truth-dir" "$WIDE_FOV_LOGGED_INFO_DIR")
                echo "Using wide-FOV recorded metadata: $WIDE_FOV_LOGGED_INFO_DIR"
            fi
        elif [ "$USE_STENCIL_GT_POSE" -eq 1 ] || has_extra_arg "--use-wide-fov-ground-truth"; then
            generate_wide_fov_ground_truth_pose_records "$WIDE_FOV_USAGE_DIR"
            if ! has_extra_arg "--wide-fov-ground-truth-dir" && ! has_extra_arg "--wide-fov-ground-truth-pose-records"; then
                RUN_PARAMS+=("--wide-fov-ground-truth-dir" "$WIDE_FOV_LOGGED_INFO_DIR")
                echo "Using wide-FOV ground-truth metadata: $WIDE_FOV_LOGGED_INFO_DIR"
            fi
        fi

        if [ "$USE_RECORDED_TEXEL_USAGE" -eq 1 ]; then
            if ! has_extra_arg "--wide-fov-texel-usage-mask-dir"; then
                RUN_PARAMS+=("--wide-fov-texel-usage-mask-dir" "$WIDE_FOV_USAGE_DIR/widefov_texel_usage")
                echo "Using wide-FOV texel usage masks: $WIDE_FOV_USAGE_DIR/widefov_texel_usage"
            fi
        fi
    fi

    mkdir -p "$LOGGED_INFO_DIR"
    if [ "$DUMP_OUTPUT" -eq 1 ]; then
        mkdir -p "$DST_DIR"
        find "$DST_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
        mkdir -p "$LOGGED_INFO_DIR"
        RUN_PARAMS+=("--save-images" "--output-path" "$DST_DIR")
    fi
    echo "Running build for scene: $SCENE"
    # $APP --size 1920x1080 --scene ../assets/scenes/$SCENE.json --save-images --camera-path ../assets/paths/$SCENE\_path.txt --output-path $DST_DIR "${PARAMS[@]}" "/media/csl-wanhanglu/SSD/quasarOutput/TestScript/predictedPoseVerseRenderPose/$SCENE/pose_differences.csv" &> $DST_DIR/build_log.txt
        (
            cd "$APP_DIR" || exit 1
            "${APP_RUNNER[@]}" "$APP_BIN" --size 1920x1080 --scene ../assets/scenes/$SCENE.json --camera-path ../assets/paths/${SCENE}_path.txt "${RUN_PARAMS[@]}" "${EXTRA_APP_ARGS[@]}"
        ) &> "$BUILD_LOG"
    if [ "$2" == "quasar" ]; then
        generate_wide_fov_ground_truth_pose_records "$DST_DIR"
    fi
    organize_logged_info "$DST_DIR"
    # # Add for pose dumping
    # mkdir -p "/media/csl-wanhanglu/SSD/quasarOutput/TestScript/predictedPoseVerseRenderPoseNoSmooth/$SCENE"
    # mv predicted_poses.csv "/media/csl-wanhanglu/SSD/quasarOutput/TestScript/predictedPoseVerseRenderPoseNoSmooth/$SCENE/predicted_poses.csv"
    # mv camera_poses.csv "/media/csl-wanhanglu/SSD/quasarOutput/TestScript/predictedPoseVerseRenderPoseNoSmooth/$SCENE/camera_poses.csv"
    echo "Completed build for scene: $SCENE"
done
