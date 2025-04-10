import numpy as np
import cv2 as cv

# The given video and calibration data
video_file = 'data/newchessboard.mp4'
K = np.array([[483.63276949,   0,         489.78786419],
              [0,         482.20222809, 258.33907507],
              [0,           0,           1]])
dist_coeff = np.array([-0.0330968, 0.1257044, 0.00088757, 0.00367228, -0.11900001])
# 이전에 수행한 결과가 적합하지 않아서, 다시 n select를 40번 수행하여 나온 calibration 결과를 사용함
board_pattern = (8, 6)
board_cellsize = 0.025
board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

# Open a video
video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given input, ' + video_file

# Prepare a 3D box for simple AR
# 사다리꼴 형 사각기둥 모양으로 변형 : x축 방향으로 늘어나고 높이는 2를 유지하고, 윗면의 면적이 더 넓어지도록 변형
box_lower = board_cellsize * np.array([[4, 3,  0],   # 왼쪽 아래
                                       [5, 3,  0],   # 오른쪽 아래
                                       [4, 4,  0],  # 왼쪽 위
                                       [5, 4,  0]])  # 오른쪽 위

box_upper = board_cellsize * np.array([[6, 2, -2], # 왼쪽 아래
                                       [7, 2, -2], # 오른쪽 아래
                                       [6, 5, -2], # 왼쪽 위
                                       [7, 5, -2]])  # 오른쪽 위


# Prepare 3D points on a chessboard
obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

# Run pose estimation
while True:
    # Read an image from the video
    valid, img = video.read()
    if not valid:
        break

    # Estimate the camera pose
    success, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
    if success:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # Draw the box on the image
        line_lower, _ = cv.projectPoints(box_lower, rvec, tvec, K, dist_coeff)
        line_upper, _ = cv.projectPoints(box_upper, rvec, tvec, K, dist_coeff)
        cv.polylines(img, [np.int32(line_lower)], True, (255, 0, 0), 2)
        cv.polylines(img, [np.int32(line_upper)], True, (0, 0, 255), 2)
        for b, t in zip(line_lower, line_upper):
            cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 2)

        # Print the camera position
        R, _ = cv.Rodrigues(rvec) # Alternative) `scipy.spatial.transform.Rotation`
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))   

    # Show the image and process the key event
    cv.imshow('Pose Estimation (Chessboard)', img)
    key = cv.waitKey(10)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27: # ESC
        break

video.release()
cv.destroyAllWindows()