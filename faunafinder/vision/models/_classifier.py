import json
import torch
from torchvision import transforms


def transform_input(img, size=(224, 224)):
    # TODO: add coercion to required format for various img types
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    return transform(img)


class Classifier:
    def __init__(self, model_path, label_json_path, error_threshold=0.5):
        # Load the trained models from the specified models path
        self.model = torch.load(model_path)
        # Set the models to eval mode
        self.model.eval()

        # Load the class labels from the specified json file
        with open(label_json_path, 'r') as f:
            self.class_labels = json.load(f)

        # Set the error threshold for determining the most likely class
        self.error_threshold = error_threshold

    def predict(self, input_data):
        # unsqueeze single image for the batch dimension
        input_tensor = input_data.unsqueeze(0)
        # Make a prediction using the models
        output = self.model(input_tensor)
        # convert logit to probability
        output = torch.nn.functional.softmax(output, dim=1)
        # Get the most likely class given the error threshold
        most_likely_class = self.get_most_likely_class(output, self.error_threshold)
        # Return the most likely class and the raw prediction output
        return output, most_likely_class

    def get_most_likely_class(self, output, error_threshold):
        # Get the class with the highest probability
        _, predicted_class = output.max(dim=1)
        # Get the probability of the predicted class
        predicted_prob = output[:, predicted_class]
        # Calculate the error as the difference between the predicted probability and 1.0
        error = 1.0 - predicted_prob
        # If the error is below the threshold, return the predicted class label
        if error < error_threshold:
            return self.class_labels[str(predicted_class.item())]
        # If the error is above the threshold, return None
        else:
            return None
