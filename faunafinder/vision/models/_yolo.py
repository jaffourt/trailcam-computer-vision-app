import torch
from PIL import Image


class ObjectDetector:

    def __init__(self):
        # Load the pretrained yolov5 from torch hub
        torch.hub.set_dir("trained_models")
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        # Set the models to eval mode
        self.model.eval()

    @staticmethod
    def image_from_path(image_path):
        return Image.open(image_path).convert("RGB")

    @staticmethod
    def crop_subset(image, rectangle):
        x0, x1, y0, y1 = rectangle
        return image.crop((x0.item(), x1.item(), y0.item(), y1.item()))

    def predict(self, x):
        return self.model(x)

    def detect_objects(self, image):
        output = self.predict(image)
        for detection in output.pred[0]:
            yield detection
