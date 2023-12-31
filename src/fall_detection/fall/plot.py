import io
import requests
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


class PoseClassificationVisualizer(object):
    """Keeps track of claassifcations for every frame and renders them."""

    def __init__(
        self,
        class_name,
        plot_location_x=0.05,
        plot_location_y=0.05,
        plot_max_width=0.8,
        plot_max_height=0.8,
        plot_figsize=(9, 4),
        plot_x_max=None,
        plot_y_max=None,
        detector_location_x=0.05,
        detector_location_y=0.25,
        detector_font_color="red",
        detector_font_size=0.05,
    ):
        self._class_name = class_name
        self._plot_location_x = plot_location_x
        self._plot_location_y = plot_location_y
        self._plot_max_width = plot_max_width
        self._plot_max_height = plot_max_height
        self._plot_figsize = plot_figsize
        self._plot_x_max = plot_x_max
        self._plot_y_max = plot_y_max
        self._detector_location_x = detector_location_x
        self._detector_location_y = detector_location_y
        self._detector_font_color = detector_font_color
        self._detector_font_size = detector_font_size

        self._detector_font = None
        self._pose_classification_history = []
        self._pose_classification_filtered_history = []

    def __call__(
        self,
        frame,
        pose_classification,
        pose_classification_filtered,
        detector_state,
    ):
        """Renders pose classifcation and counter until given frame."""
        # Extend classification history.
        self._pose_classification_history.append(pose_classification)
        self._pose_classification_filtered_history.append(pose_classification_filtered)

        # Output frame with classification plot and counter.
        output_img = Image.fromarray(frame)

        output_width = output_img.size[0]
        output_height = output_img.size[1]

        # Draw the plot.
        img = self._plot_classification_history(output_width, output_height)

        img.thumbnail(
            (
                int(output_width * self._plot_max_width),
                int(output_height * self._plot_max_height),
            ),
            Image.Resampling.LANCZOS,
        )
        output_img.paste(
            img,
            (
                int(output_width * self._plot_location_x),
                int(output_height * self._plot_location_y),
            ),
        )

        # Draw the count.
        output_img_draw = ImageDraw.Draw(output_img)

        if self._detector_font is None:
            font_size = int(output_height * self._detector_font_size)
            font_request = requests.get(
                "https://github.com/googlefonts/roboto/blob/main/src/hinted/Roboto-Regular.ttf?raw=true",
                allow_redirects=True,
            )
            self._detector_font = ImageFont.truetype(
                io.BytesIO(font_request.content), size=font_size
            )
        output_img_draw.text(
            (
                output_width * self._detector_location_x,
                output_height * self._detector_location_y,
            ),
            f"{self._class_name}: " + str(detector_state),
            fill=self._detector_font_color,
            font=self._detector_font,
        )

        return output_img

    def _plot_classification_history(self, output_width, output_height):
        fig = plt.figure(figsize=self._plot_figsize)

        for classification_history in [
            self._pose_classification_history,
            self._pose_classification_filtered_history,
        ]:
            y = []
            for classification in classification_history:
                if classification is None:
                    y.append(None)
                elif self._class_name in classification:
                    y.append(classification[self._class_name])
                else:
                    y.append(0)
            plt.plot(y, linewidth=7)

        plt.grid(axis="y", alpha=0.75)
        plt.xlabel("Frame")
        plt.ylabel("Confidence")
        plt.title("Classification history for `{}`".format(self._class_name))
        # plt.legend(loc="upper right")

        if self._plot_y_max is not None:
            plt.ylim(top=self._plot_y_max)
        if self._plot_x_max is not None:
            plt.xlim(right=self._plot_x_max)

        # Convert plot to image.
        buf = io.BytesIO()
        dpi = min(
            output_width * self._plot_max_width / float(self._plot_figsize[0]),
            output_height * self._plot_max_height / float(self._plot_figsize[1]),
        )
        fig.savefig(buf, dpi=dpi)
        buf.seek(0)
        img = Image.open(buf)
        plt.close()

        return img


def plot_fall_text(image, fall):
    text_color = (0, 0, 255) if fall else (0, 255, 0)
    fall_text = "Fall" if fall else "No Fall"

    _, image_width = image.shape[:2] 
    fall_text_position = (image_width - 150, 30)

    return cv2.putText(
        image, fall_text, fall_text_position,
        cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

