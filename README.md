# AISL-KICKOFF-pothole

- YOLOv5
    - [YOLOv5 설치](https://www.notion.so/OpenCV-584e6a2b38fc4d2c92381c1fc5433e91)
    - [Train Custom Data](https://www.notion.so/OpenCV-584e6a2b38fc4d2c92381c1fc5433e91)
    - [Training](https://www.notion.so/OpenCV-584e6a2b38fc4d2c92381c1fc5433e91)
    
    <aside>
    💡 사용자 정의 개체를 감지하는 법 ! (포트홀)
    → 사용자 정의 YOLO 모델 생성
    
    1. 포트홀 이미지 데이터셋 생성
    2. 해당 데이터셋에서 YOLO 모델 훈련
    
    </aside>
    

---

## YOLOv5 설치

[YOLOv5 Documentation](https://docs.ultralytics.com/)

### 1. 깃헙 클론

[https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)

```python
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # 필요한 패키지 설치
```

### 2. Object Detection

- clone 하지 않은 경우

PyTorch Hub에서 직접 실행

```python
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

# Images
img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
```

- clone 한 경우

실행 결과는 `./runs/detect`에 저장

```python
python detect.py --source # OPTION
													0  # 연결된 webcam에서 실시간으로 detect
                          파일이름.jpg  # image
                          파일이름.mp4  # video
                          screen  # screenshot
                          디렉터리이름/  # directory
                          'path/*.jpg'  # glob
                          'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                          'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

- Training

YOLOv5 COCO 결과 재현

models, datasets는 최신 YOLOv5에서 자동으로 다운

YOLOv5n/s/m/l/x의 훈련 시간은 V100 GPU에서 1/2/4/6/8일

```python
python train.py --data coco.yaml --cfg yolov5n.yaml --weights '' --batch-size 128
                                       yolov5s                                64
                                       yolov5m                                40
                                       yolov5l                                24
                                       yolov5x                                16
```

---

## Train Custom Data

- 참고자료
    
    [Train Custom Data · ultralytics/yolov5 Wiki](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
    
    [Train Custom Data 📌](https://docs.ultralytics.com/tutorials/train-custom-datasets/)
    

### 1. Create Dataset

- roboflow 사용 → 자동으로 레이블 지정, 준비, 호스팅
    - 이미지 수집
    - 라벨 생성
    - YOLOv5를 위한 데이터셋 준비

- 수동으로 준비
    - dataset.yaml 생성
    - 라벨 생성
    - 디렉터리 정의
    

### 2. 모델 선택

- YOLOv5n
- YOLOv5s → 추천
- YOLOv5m
- YOLOv5l
- YOLOv5x

### 3. Train

```python
# Train YOLOv5s on COCO128 for 3 epochs
$ python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5s.pt
```

`--weights yolov5s.pt`(recommended)

 `--weights '' --cfg yolov5s.yaml`(not recommended)

 `--cache ram`or `--cache disk`: 속도를 높이려면 추가

local dataset에서 train (google drive는 느림)

모든 실행결과는 `runs/train/`에 `runs/train/exp2`, `runs/train/exp3`이런 식으로 저장됨

### 4. Visualize

- ClearML

```python
pip install clearml

clearml-init # ClearML 서버에 연결하기 위해 실행
```

---

## YOLOv5 Training

1. YOLOv5 git clone

```python
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # 필요한 패키지 설치
```

2. dataset

[yolo v5 pothole detection median blur Object Detection Dataset by Parth Choksi](https://universe.roboflow.com/parth-choksi/yolo-v5-pothole-detection-median-blur)

- images
- labels


3. coco128.yaml 파일 수정

```python
train: ../pothole_blur_dataset/train/images # 데이터셋 train 경로
val: ../pothole_blur_dataset/valid/images # 데이터셋 val 경로

nc: 1 # 클래스 -> pothole 1개
names: ['Potholes']
```



4. yolov5s.yaml 파일 수정

```python
# Parameters
nc: 1  # number of classes
```

클래스 개수만 수정


5. YOLOv5 확인

```python
python detect.py --source # OPTION
													0  # 연결된 webcam에서 실시간으로 detect
                          파일이름.jpg  # image
                          파일이름.mp4  # video
                          screen  # screenshot
                          디렉터리이름/  # directory
                          'path/*.jpg'  # glob
                          'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                          'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

6. YOLOv5 Training

```python
python train.py --data data/coco128.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt --img 640 --batch 16 --epochs 3
```

7. test

```python
python detect.py --source .\data\images\2.jpg --weights .\runs\train\exp_best\weights\best.pt
```

—source : 이미지

—weights : pre-trained 모델

- 새로운 데이터셋

    
    ```python
    train: ../pothole_dataset/train
    val: ../pothole_dataset/valid
    
    nc: 1
    names: ['Potholes']
    ```
    
    ```python
    python train.py --data .\data\coco128.yaml --cfg .\models\yolov5s.yaml --weights yolov5s.pt --img 640 --batch 16 --epochs 3
    ```
    

test 결과 한 폴더에 저장될 수 있도록 detect.py 코드 수정

```python
# Directories
    save_dir = Path(project)/'result'
    save_dir.mkdir(parents=True, exist_ok=True)

    #save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    #(save_dir / 'labels').mkdir(parents=True, exist_ok=True)  # make dir
```
