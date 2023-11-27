import torchvision
import torchvision.models.detection.faster_rcnn as frcnn
def create_model(num_classes):
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=frcnn.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = frcnn.FastRCNNPredictor(in_features, num_classes) 
    return model