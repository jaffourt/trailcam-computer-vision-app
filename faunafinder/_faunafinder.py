from typing import Iterable, Tuple
from PIL import ImageDraw, Image, ImageFont

from .vision.models import Classifier, ObjectDetector, transform_input
from .webcam import Webcam


class FaunaFinder:
    """
    A class for capturing webcam images and detecting and classifying objects in images using an object
    detector and classifier.

    Attributes:

    _classifier (Classifier): The object classifier to use.
    _object_detector (ObjectDetector): The object detector to use.
    _webcam (Webcam): The webcam instance to use.

    Public API Methods:

    __call__ : Uses the object detector and classifier to detect and classify objects in an image
    from the webcam. Returns the image with bounding boxes and labels for detected objects.

    configure_webcam : Attempts to configure a webcam instance from user input.

    draw_bbox : Draws a bounding box on an image.

    find_fauna : Detects and classifies objects in an image using the object detector and classifier.
    Returns tuples containing the classification and bounding box rectangle for each object.
    """

    def __init__(self, model_path: str,
                 label_json_path: str,
                 error_threshold: float = 0.1,
                 webcam: Webcam = None):
        """
        Initializes the FaunaFinder with a classifier, object detector, and webcam instance.

        Parameters:
        model_path (str): The path to the model to use for object classification.
        label_json_path (str): The path to the JSON file containing labels for the object classification model.
        error_threshold (float, optional): The maximum acceptable error threshold for object classification. Must be
            a value between 0 and 1, a value of 1 means there is no error threshold. Defaults to 0.1.
        webcam (Webcam, optional): The webcam instance to use. If not provided, one can be created from user input.
        """
        self._classifier = Classifier(
            model_path=model_path,
            label_json_path=label_json_path,
            error_threshold=error_threshold)
        self._object_detector = ObjectDetector()
        self._webcam = webcam

        # try to load a common font for displaying detections
        self._label_font = None
        try:
            self._label_font = ImageFont.truetype("Arial.ttf", size=42)
        except (OSError, Exception):
            # OSError is expected if font is not found
            # for any other exception we can still use defaults
            self._label_font = ImageFont.load_default()

    def __call__(self):
        # get webcam data
        if self._webcam is None:
            raise Exception("Webcam instance is None.")
        raw_image = self._webcam.get_frame()
        detection_data = self.find_fauna(raw_image)

        updated_image = raw_image.copy()
        for animal, bbox in detection_data:
            if animal is not None:
                # add bbox around object, and add animal label
                # return the image with objects and labels
                self.draw_bbox(updated_image, bbox, label=animal)

        return updated_image

    def __del__(self):
        """Ensure that the webcam is released on deletion."""
        if self._webcam is not None:
            self._webcam.delete()

    def configure_webcam(self):
        """Attempt to configure a webcam instance from user input"""
        self._webcam = Webcam.from_user_input()

    def create_webcam_instance(self, index: int):
        """Create a webcam instance from user input"""
        self._webcam = Webcam(index)

    def draw_bbox(self, image, bbox: Tuple[float],
                  label: str = None,
                  color="white",
                  width=4,
                  color_map: dict = None):
        """
        Draws a bounding box on an image.

        Parameters:
        image (Image): The image to draw the bounding box on.
        bbox (Tuple[float]): A tuple representing the bounding box as (left, top, right, bottom).
        label (str, optional): The label to draw above the bounding box. Defaults to None.
        color (str, optional): The color of the bounding box and label. Defaults to "white".
        width (int, optional): The width of the bounding box line. Defaults to 4.
        color_map (dict, optional): A mapping of labels to colors. If provided, the color of the bounding
            box will be determined by this mapping for the given label. Defaults to None.
        """
        draw = ImageDraw.Draw(image)
        if color_map is not None:
            color = color_map[label]
        draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline=color, width=width)
        if label is not None:
            draw.text((bbox[0], bbox[1]), label, align="left", fill="black", font=self._label_font)

    def find_fauna(self, image: Image) -> Iterable[Tuple]:
        """
        This method takes in an image and uses the object detector to detect objects within the image.
        It then classifies each object using the classifier and returns a tuple containing the classification
        and bounding box rectangle for each object.

        Parameters:
        image: A PIL.Image to detect and classify objects within (in RGB format).

        Returns:
        An iterator yielding tuples containing the classification and bounding box rectangle for each object.
        """
        detected_objects = self._object_detector.detect_objects(image)

        # classify objects
        for obj in detected_objects:
            # bounding box of detected object
            rectangle = obj[0:4]

            # get data inside bbox and classify
            x = self._object_detector.crop_subset(image, rectangle)
            _, classification = self._classifier.predict(transform_input(x))

            yield classification, rectangle
