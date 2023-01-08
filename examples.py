"""An example of interfacing with FaunaFinder's public API"""
from faunafinder import FaunaFinder
from faunafinder.webcam import Webcam
from PIL import Image


# #################################################################################
# manually creating a webcam instance, and detecting images from the webcam
# #################################################################################
def example1():
    webcam = Webcam(index=0)
    finder = FaunaFinder(model_path="trained_models/finetuned_vg_166_2023-01-05 21:49:42.657696.pt",
                         label_json_path="data/class_labels.json", error_threshold=0.2,
                         webcam=webcam)

    # checks the webcam input and detects objects then updates the image with detections
    detected = finder()
    detected.show()

    # releases the webcam
    del finder


# #################################################################################
# Using the FaunaFinder.configure_webcam() method for creating a webcam instance
# #################################################################################
def example2():
    finder = FaunaFinder(model_path="trained_models/finetuned_vg_166_2023-01-05 21:49:42.657696.pt",
                         label_json_path="data/class_labels.json", error_threshold=0.2)

    # waits for user to supply the index of the desired camera
    finder.configure_webcam()

    # capture webcam input and run through pipeline
    detected = finder()
    detected.show()

    # releases the webcam
    del finder


# #################################################################################
# Using FaunaFinder.find_fauna() method to detect objects in user provided image
# #################################################################################
def example3():
    finder = FaunaFinder(model_path="trained_models/finetuned_vg_166_2023-01-05 21:49:42.657696.pt",
                         label_json_path="data/class_labels.json", error_threshold=0.2)

    image = Image.open("data/squirrel_and_bird.png").convert("RGB")
    for animal, bbox in finder.find_fauna(image):
        # animals detected by classifier are "deer", "bird", "squirrel"
        if animal is not None:
            finder.draw_bbox(image, bbox, label=animal, color_map={
                "squirrel": "red",
                "bird": "blue",
                "deer": "white",
            })
    image.show()


example2()
