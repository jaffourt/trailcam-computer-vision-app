import cv2
from PIL import Image


class Webcam:
    """A class for capturing video from a webcam using OpenCV.

    Attributes:
        index (int): The index of the webcam to use.
        cap (cv2.VideoCapture): The `cv2.VideoCapture` object for the webcam.
    """

    def __init__(self, index=0):
        """Initializes the `Webcam` object.

        Args:
            index (int, optional): The index of the webcam to use. Defaults to 0.
        """
        self.index = index
        self.cap = cv2.VideoCapture(index)

    def __del__(self):
        self.delete()

    def delete(self):
        """Releases the webcam when the `Webcam` object is deleted."""
        self.cap.release()

    @staticmethod
    def get_webcams():
        available_webcams = {}
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_webcams[i] = cap.getBackendName()
            cap.release()
        return available_webcams

    @classmethod
    def from_user_input(cls):
        """Creates a new `Webcam` instance based on user input.

        Lists the available webcams and prompts the user to select a webcam. If the selected
        webcam is available, creates a new `Webcam` instance with the selected index and returns
        it. If the selected webcam is not available or no webcams are available, raises a
        `ValueError`.

        Returns:
            Webcam: The new `Webcam` instance.

        Raises:
            ValueError: If the selected webcam is not available or no webcams are available.
        """
        # Check the available webcams
        available_webcams = cls.get_webcams()

        # Prompt the user to select a webcam
        if available_webcams:
            print("\nAVAILABLE WEBCAMS\n")
            for i, name in available_webcams.items():
                print(f'\tWebcam {i}: {name}')
            index = int(input('\nSelect a webcam: '))
            if index in available_webcams:
                return cls(index)
            else:
                raise ValueError('Invalid webcam index')
        else:
            raise ValueError('No webcams available')

    def get_frame(self):
        """Captures and returns a frame from the webcam.

        Returns:
            Image: The frame as a PIL Image object.
        """
        ret, frame = self.cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
