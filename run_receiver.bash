nsys profile \
    --trace=cuda,nvtx,opengl --opengl-gpu true -o receiver_mesh_$1 --delay 20\
    ./build/apps/quasar/receiver/qr_receiver \
    --size 1920x1080 --pose-url 192.168.0.112:54321 \
    --video-url 0.0.0.0:12345 --proxies-url 192.168.0.112:65432 \
    --camera-path ./assets/paths/robot_lab_path_trimmed.txt \
    &> receiver_mesh_$1.txt