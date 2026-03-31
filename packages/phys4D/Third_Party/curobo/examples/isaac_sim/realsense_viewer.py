











import cv2
import numpy as np
from nvblox_torch.datasets.realsense_dataset import RealsenseDataloader


def view_realsense():
    realsense_data = RealsenseDataloader(clipping_distance_m=1.0)

    try:
        while True:
            data = realsense_data.get_raw_data()
            depth_image = data[0]
            color_image = data[1]



            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=100), cv2.COLORMAP_JET
            )
            images = np.hstack((color_image, depth_colormap))

            cv2.namedWindow("Align Example", cv2.WINDOW_NORMAL)
            cv2.imshow("Align Example", images)
            key = cv2.waitKey(1)

            if key & 0xFF == ord("q") or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        realsense_data.stop_device()


if __name__ == "__main__":
    view_realsense()
