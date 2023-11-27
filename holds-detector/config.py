import torch
BATCH_SIZE = 1 
RESIZE_TO = 512 
NUM_EPOCHS = 10 
NUM_WORKERS = 2
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
TRAIN_DIR = 'data/dataset/train'
VALID_DIR = 'data/dataset/valid'
CLASSES = [
    '__background__', 'hold'
    ]
NUM_CLASSES = len(CLASSES)
OUT_DIR = 'outputs'