import numpy as np
import cv2

from mc3d_trecsim.enums import KPT_NAMES, KPT_NAMES_SHORT, KPT_NAMES_SHORTER

def paint_skeleton_on_image(image: np.ndarray, keypoints, plot_sides: bool = False):
    #Plot the skeleton and keypointsfor coco datatset
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5

    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(keypoints[(sk[0]-1), 0]), int(keypoints[(sk[0]-1), 1]))
        pos2 = (int(keypoints[(sk[1]-1), 0]), int(keypoints[(sk[1]-1), 1]))
        conf1 = keypoints[(sk[0]-1), 2]
        conf2 = keypoints[(sk[1]-1), 2]
        if conf1<0.5 or conf2<0.5:
            continue
        if pos1[0]%640 == 0 or pos1[1]%640==0 or pos1[0]<0 or pos1[1]<0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0]<0 or pos2[1]<0:
            continue
        cv2.line(image, pos1, pos2, (int(r), int(g), int(b)), thickness=2)

    for kid, kpt in enumerate(keypoints):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpt[0], kpt[1]
        if not (x_coord % image.shape[1] == 0 or y_coord % image.shape[0] == 0):
            conf = kpt[2]
            if conf < 0.5:
                continue
            cv2.circle(image, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)

            if plot_sides:
                offset = len(KPT_NAMES_SHORTER[kid])/2 * 10
                cv2.putText(image, KPT_NAMES_SHORTER[kid], (int(x_coord - offset), int(
                    y_coord)), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 0), 8, cv2.LINE_AA)
                cv2.putText(image, KPT_NAMES_SHORTER[kid], (int(x_coord - offset), int(
                    y_coord)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 4, cv2.LINE_AA)
