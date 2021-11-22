# Yolov4 training
YOLO_TYPE                   = "yolov4"

# Set train parameters 
YOLO_V4_WEIGHTS             = "Model/yolov4.weights"
TRAIN_CLASSES               = "Data/classes.txt"
TRAIN_ANNOT_PATH            = "Data/train.txt"
YOLO_COCO_CLASSES           = "Model/coco/coco.names" # Don't change this
YOLO_CUSTOM_WEIGHTS         = True

# Set Validate parameters
TEST_ANNOT_PATH             = "./Data/valid.txt"
TEST_BATCH_SIZE             = 4
TEST_INPUT_SIZE             = 416
TEST_DATA_AUG               = False
TEST_DECTECTED_IMAGE_PATH   = ""
TEST_SCORE_THRESHOLD        = 0.3
TEST_IOU_THRESHOLD          = 0.45

SAVE_CHECKPOINTS_FOLDER    = "checkpoints/"
SAVE_MODEL_NAME            = "yolov4_custom"


TRAIN_BATCH_SIZE            = 16
TRAIN_INPUT_SIZE            = 416
TRAIN_DATA_AUG              = True
TRAIN_TRANSFER              = True
TRAIN_FROM_CHECKPOINT       = False
if TRAIN_FROM_CHECKPOINT:
  CHECKPOINT_PATH = 'checkpoints/yolov4_custom'
TRAIN_LR_INIT               = 1e-4
TRAIN_LR_END                = 1e-6
TRAIN_WARMUP_EPOCHS         = 2
TRAIN_EPOCHS                = 100

# Train options
TRAIN_SAVE_BEST_ONLY        = True
TRAIN_LOAD_IMAGES_TO_RAM    = True # With True faster training, but need more RAM

# Model options
YOLO_STRIDES                = [8, 16, 32]
YOLO_IOU_LOSS_THRESH        = 0.5
YOLO_ANCHOR_PER_SCALE       = 3
YOLO_MAX_BBOX_PER_SCALE     = 100
YOLO_INPUT_SIZE             = 416
YOLO_ANCHORS                = [[[12,  16], [19,   36], [40,   28]],
                               [[36,  75], [76,   55], [72,  146]],
                               [[142,110], [192, 243], [459, 401]]]


