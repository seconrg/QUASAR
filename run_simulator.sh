# Create a list of shell scripts to run
#!/bin/bash
# List of name, {viking_village, sponza, robot_lab}
SCENES=(
    "viking_village" 
    "sponza" 
    "robot_lab"
    "sun_temple"
    "san_miguel"
)

APP=./build/apps/scene_viewer/scene_viewer
POSE_PREDICTOR=""

if [ "$2" == "quasar" ]; then
    echo "Running Quasar build script..."
    APP=./build/apps/quasar/simulator/qr_simulator
    POSE_PREDICTOR="--pose-prediction --network-latency 20.0 --network-jitter 10.0 "
elif [ "$2" == "baseline" ]; then
    echo "Running Baseline build script..."
    APP=./build/apps/scene_viewer/scene_viewer
    POSE_PREDICTOR=""
elif [ "$2" == "quadstream" ]; then
    echo "Running QuadStream build script..."
    APP=./build/apps/quadstream/simulator/qs_simulator
    POSE_PREDICTOR="--pose-prediction --network-latency 20.0 --network-jitter 10.0 "
elif [ "$2" == "meshwarp" ]; then
    echo "Running MeshWarp build script..."
    APP=./build/apps/meshwarp/simulator/mw_simulator
    POSE_PREDICTOR="--pose-prediction --network-latency 20.0 --network-jitter 10.0 "
elif [ "$2" == "atw" ]; then
    echo "Running ATW build script..."
    APP=./build/apps/atw/simulator/atw_simulator
    POSE_PREDICTOR="--pose-prediction --network-latency 20.0 --network-jitter 10.0 "
else 
    echo "Please provide a valid application name (e.g., quasar)."
    exit 1
fi

# Remove the directory if it exists
for SCENE in "${SCENES[@]}"; do
    DST_DIR=/media/csl-wanhanglu/SSD/quasarOutput/$1/$SCENE/$2$3
    rm -rf $DST_DIR
    mkdir -p $DST_DIR
    echo "Running build for scene: $SCENE"
    $APP --size 1920x1080 --scene ../assets/scenes/$SCENE.json --save-images --camera-path ../assets/paths/$SCENE\_path.txt --output-path $DST_DIR $POSE_PREDICTOR &> $DST_DIR/build_log.txt
    echo "Completed build for scene: $SCENE"
done