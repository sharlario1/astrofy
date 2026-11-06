"""MediaPipe Face Landmarker (478 pts) index groups.

Generated from mediapipe.tasks.python.vision.FaceLandmarksConnections (v1.2.2),
so they do not depend on memory. Masks use convex hulls of these groups, so
point ordering is irrelevant.
"""
FACE_OVAL = [10, 21, 54, 58, 67, 93, 103, 109, 127, 132, 136, 148, 149, 150, 152, 162, 172, 176, 234, 251, 284, 288, 297, 323, 332, 338, 356, 361, 365, 377, 378, 379, 389, 397, 400, 454]
LEFT_EYE = [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466]
RIGHT_EYE = [7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246]
LEFT_EYEBROW = [276, 282, 283, 285, 293, 295, 296, 300, 334, 336]
RIGHT_EYEBROW = [46, 52, 53, 55, 63, 65, 66, 70, 105, 107]
LIPS = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415]
# Iris: centre + 4 ring points (right eye = subject's right = 468.., left = 473..)
RIGHT_IRIS_CENTER, RIGHT_IRIS_RING = 468, [469, 470, 471, 472]
LEFT_IRIS_CENTER, LEFT_IRIS_RING = 473, [474, 475, 476, 477]
N_LANDMARKS = 478
