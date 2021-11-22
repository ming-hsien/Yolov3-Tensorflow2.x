# Yolov4-Tensorflow2.x

## Training custom model

This is a repo for yolov4 using tensorflow2.x
You can inference and training your own model.

### Steps
* `git clone https://github.com/ming-hsien/Yolov4-Tensorflow2.x.git`
* Modify the parameter in `yolov4/configs.py`
* Download yolov4 model
```
  cd Model
  wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
```
* Make your train and valid Dataset as `Data/train.txt` and `Data/valid.txt`
* Modify label name in `Data/classes.txt`

__Example__

|line|label|
|:-:|:-:|
| 0 |cat|
| 1 |dog|
| 2 |pig|
| X |...|

* `python train.py`

## Testing custom model
See `detect.py`
