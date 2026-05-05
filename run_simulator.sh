#!/bin/bash
SCENES=(
    "viking_village" 
    "sponza" 
    # "robot_lab"
    "sun_temple"
    "san_miguel"
)

APP=./apps/scene_viewer/scene_viewer
PARAMS=()
EXTRA_PARAMS=("${@:4}")

if [ "$2" == "quasar" ]; then
    echo "Running Quasar build script..."
    APP=./apps/quasar/simulator/qr_simulator
    if [ "${#EXTRA_PARAMS[@]}" -gt 0 ] && [[ "${EXTRA_PARAMS[0]}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        PARAMS+=("--remote-fov-wide" "${EXTRA_PARAMS[0]}")
        EXTRA_PARAMS=("${EXTRA_PARAMS[@]:1}")
    fi
    if [ "${#EXTRA_PARAMS[@]}" -gt 0 ] && [ "${EXTRA_PARAMS[0]}" == "dump" ]; then
        # This wrapper already passes --save-images below; keep legacy invocations working.
        EXTRA_PARAMS=("${EXTRA_PARAMS[@]:1}")
    fi
    PARAMS+=("--pose-prediction" "--network-latency" "20.0" "--network-jitter" "0.0")
    PARAMS+=("--view-size" "0.25")
    # PARAMS+=("--E-path")
elif [ "$2" == "baseline" ]; then
    echo "Running Baseline build script..."
    APP=./apps/scene_viewer/scene_viewer
    POSE_PREDICTOR=()
elif [ "$2" == "quadstream" ]; then
    echo "Running QuadStream build script..."
    APP=./apps/quadstream/simulator/qs_simulator
    PARAMS+=("--pose-prediction" "--pose-smoothing" "--network-latency" "20.0" "--network-jitter" "0.0")
elif [ "$2" == "meshwarp" ]; then
    echo "Running MeshWarp build script..."
    APP=./apps/meshwarp/simulator/mw_simulator
    PARAMS+=("--pose-prediction" "--pose-smoothing" "--network-latency" "20.0" "--network-jitter" "0.0")
elif [ "$2" == "atw" ]; then
    echo "Running ATW build script..."
    APP=./apps/atw/simulator/atw_simulator
    PARAMS+=("--pose-prediction" "--pose-smoothing" "--network-latency" "20.0" "--network-jitter" "0.0")
elif [ "$2" == "hybrid" ]; then
    echo "Running Hybrid build script..."
    APP=./apps/hybrid/simulator/hybrid_simulator
    PARAMS+=("--pose-prediction" "--network-latency" "20.0" "--network-jitter" "0.0")
    # PARAMS+=("--E-path")
else
    echo "Please provide a valid application name (e.g., quasar)."
    exit 1
fi

# Remove the directory if it exists
for SCENE in "${SCENES[@]}"; do
    DST_DIR=/media/wuhaolu/c54fff3f-cab5-4dcf-94c3-c83855e5a9bd/quasarOutput/$1/$SCENE/$2$3
    rm -rf $DST_DIR
    mkdir -p $DST_DIR
    echo "Running build for scene: $SCENE"
    # $APP --size 1920x1080 --scene ../assets/scenes/$SCENE.json --save-images --camera-path ../assets/paths/$SCENE\_path.txt --output-path $DST_DIR "${PARAMS[@]}" "/media/csl-wanhanglu/SSD/quasarOutput/TestScript/predictedPoseVerseRenderPose/$SCENE/pose_differences.csv" &> $DST_DIR/build_log.txt
        $APP --size 1920x1080 --scene ../assets/scenes/$SCENE.json --save-images --camera-path ../assets/paths/$SCENE\_path.txt --output-path $DST_DIR "${PARAMS[@]}" "${EXTRA_PARAMS[@]}" &> $DST_DIR/build_log.txt
    # # Add for pose dumping
    # mkdir -p "/media/csl-wanhanglu/SSD/quasarOutput/TestScript/predictedPoseVerseRenderPoseNoSmooth/$SCENE"
    # mv predicted_poses.csv "/media/csl-wanhanglu/SSD/quasarOutput/TestScript/predictedPoseVerseRenderPoseNoSmooth/$SCENE/predicted_poses.csv"
    # mv camera_poses.csv "/media/csl-wanhanglu/SSD/quasarOutput/TestScript/predictedPoseVerseRenderPoseNoSmooth/$SCENE/camera_poses.csv"
    echo "Completed build for scene: $SCENE"
done
