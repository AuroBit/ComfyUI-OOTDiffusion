import numpy as np

# from pytorch_lightning import seed_everything
from .annotator.util import resize_image, HWC3
from .annotator.openpose import OpenposeDetector

from PIL import Image
import torch

class OpenPose:
    def __init__(self, body_model_path: str, *, device):
        self.preprocessor = OpenposeDetector(body_model_path, device=device)

    def __call__(self, input_image, resolution=384):
        if isinstance(input_image, Image.Image):
            input_image = np.asarray(input_image)
        elif type(input_image) == str:
            input_image = np.asarray(Image.open(input_image))
        else:
            raise ValueError
        with torch.no_grad():
            input_image = HWC3(input_image)
            input_image = resize_image(input_image, resolution)
            H, W, C = input_image.shape
            assert (H == 512 and W == 384), 'Incorrect input image shape'
            pose, detected_map = self.preprocessor(input_image, hand_and_face=False)

            candidate = pose['bodies']['candidate']
            subset = pose['bodies']['subset'][0][:18]
            for i in range(18):
                if subset[i] == -1:
                    candidate.insert(i, [0, 0])
                    for j in range(i, 18):
                        if(subset[j]) != -1:
                            subset[j] += 1
                elif subset[i] != i:
                    candidate.pop(i)
                    for j in range(i, 18):
                        if(subset[j]) != -1:
                            subset[j] -= 1

            candidate = candidate[:18]

            for i in range(18):
                candidate[i][0] *= 384
                candidate[i][1] *= 512

            keypoints = {"pose_keypoints_2d": candidate}
            # with open("/home/aigc/ProjectVTON/OpenPose/keypoints/keypoints.json", "w") as f:
            #     json.dump(keypoints, f)
            #
            # # print(candidate)
            # output_image = cv2.resize(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB), (768, 1024))
            # cv2.imwrite('/home/aigc/ProjectVTON/OpenPose/keypoints/out_pose.jpg', output_image)

        return keypoints


if __name__ == '__main__':

    model = OpenPose()
    model('./images/bad_model.jpg')
