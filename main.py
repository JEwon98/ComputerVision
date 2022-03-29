import cv2
import numpy as np
from matplotlib import pyplot as plt

patch = [[[[0 for _ in range(9)] for _ in range(9)] for _ in range(4)] for _ in range(2)]
hist = [0 for _ in range(8)]
click = 0


def mouse_event(event, x, y, flags, param):
    global click
    patch_size = 4
    if event == cv2.EVENT_FLAG_LBUTTON:
        print("x =", x, ", y = ", y, ", click = ", click)
        patch[click // 4 % 2][click % 4] = param[y-patch_size-1:y+patch_size, x-patch_size-1:x+patch_size]

        hist[click] = cv2.calcHist([patch[click // 4 % 2][click % 4]], [0], None, [16], [0, 256])
        plt.hist(patch[click // 4 % 2][click % 4].ravel(), 16, [0, 256])
        plt.show()
        # print(hist[click])
        xgrad = np.array(param[y-patch_size-1:y+patch_size, x-patch_size-2:x+patch_size-1]) - np.array(param[y-patch_size-1:y+patch_size, x-patch_size:x+patch_size+1])
        ygrad = np.array(param[y-patch_size-2:y+patch_size-1, x-patch_size-1:x+patch_size]) - np.array(param[y-patch_size:y+patch_size+1, x-patch_size-1:x+patch_size])
        grad_size = xgrad**2 + ygrad**2
        grad_size = np.sqrt(grad_size)
        grad_direction = np.arctan2(xgrad,ygrad)
        print("grad_size")
        print(grad_size)
        print("grad_direction")
        print(grad_direction)
        # for i in range(9):
        #     for j in range(9):
        #         print(patch[click // 4 % 2][click % 4][i][j], end=" ")
        #     print()
        # print("----------------------------")
        click += 1
        cv2.rectangle(param, (x-patch_size, y-patch_size), (x+patch_size, y+patch_size), 1)


src = np.full((500, 500, 3), 255, dtype=np.uint8)
src1 = cv2.imread("1st.jpg", cv2.IMREAD_GRAYSCALE)
src2 = cv2.imread("2nd.jpg", cv2.IMREAD_GRAYSCALE)
# height, width = src1.shape
# print(height, width)
# print(src1[0][0])
# cv2.imshow("draw", src1)
# cv2.setMouseCallback("draw", mouse_event, src1)
# while(True):
#     cv2.imshow("draw",src1)
#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:
#         break
# cv2.destroyAllWindows()

# resize
dst1 = cv2.resize(src1, dsize=(0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
dst2 = cv2.resize(src2, dsize=(0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
addh = cv2.hconcat([dst1, dst2])
cv2.imshow("1st.jpg", dst1)
cv2.imshow("2nd.jpg", dst2)
cv2.setMouseCallback("1st.jpg", mouse_event, dst1)
cv2.setMouseCallback("2nd.jpg", mouse_event, dst2)
while(True):
    cv2.imshow("1st.jpg", dst1)
    cv2.imshow("2nd.jpg", dst2)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()