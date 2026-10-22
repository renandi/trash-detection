# Trash Detection with YOLO

This repository implements an API using FastAPI and a front-end using Streamlit to detect trash in images.

The dataset used for training is the [TACO Detect](https://platform.ultralytics.com/ultralytics/datasets/taco-detect).

This project serves as the final graded assignment for the "Machine Learning and Computer Vision" course from SENAI's LAB365 for the SCTEC program.

## Setting up my environment

To train the model in my RADEON 9600 XT 16GB GPU I had to take into account some additional requirements:

- ROCm Ecossystem requires Python 3.12
- Install ROCm libraries

```ps1
uv venv --python 3.12 .venv
```

```ps1
uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "rocm[libraries,device-gfx1200]==10.1.0"
```

Then run this on the root directory to build the container (make sure Docker Engine is configured to use WSL2):

```ps1
docker build -t taco .
```

## Training and fine tuning the model


Then run the container and YOLO with:

```ps1
docker run -it --rm `
  --device=/dev/dxg `
  -v /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so `
  -v "C:\Users\rncrn\python-projects\sctec\taco:/workspace" `
  -w /workspace `
  --cap-add=SYS_PTRACE `
  --security-opt seccomp=unconfined `
  --ipc=host `
  --shm-size 8G `
  taco-detect yolo detect train data=/workspace/taco-detect.ndjson model=yolov8n.pt epochs=30 imgsz=640 batch=16 workers=2 device=0 name='train-v8n'
```
### Refine the model

To run a better model (YOLO26s) with greater image size (1280) and without mosaic on the last 10 epochs:

```bash
yolo detect train data=/workspace/taco-detect.ndjson model=yolo26s.pt epochs=50 imgsz=1280 batch=8 workers=2 mosaic=0.5 close_mosaic=10 device=0 name='train-26s-1280'
```

## Results analysis

Using YOLO26s the results were not satisfactory. YOLOv8n detected more objects in some images, but there were many errors. For example there were problems in detecting cigarretes or classifying bottles as plastic wrapper or cans as bottles.

This is the reason why I decided to train a better, refined model (YOLO26s) with some additional changes.

Diving deeper into this topic, I found out that YOLO uses mosaic by combining 4 images into one for training, therefore losing small details like those present in cigarretes for example. So I disabled the mosaic for the last 10 epochs.

OR

Finally for the biggest model I can execute on my GPU:

```bash
yolo detect train data=/workspace/taco-detect.ndjson model=yolov8l.pt epochs=50 imgsz=1280 batch=8 workers=2 device=0 mosaic=0.5 close_mosaic=10 name='train-v8l-1280'
```