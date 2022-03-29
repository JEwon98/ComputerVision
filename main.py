import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

patch = [[[[0 for _ in range(9)] for _ in range(9)] for _ in range(4)] for _ in range(2)]
#histogram of Color
hoC = np.zeros(shape=(2,4,9,9))
#histogram of Gradient
hoG = np.zeros(shape=(2,4,9,9))
#Histogram of Size
hoS = np.zeros(shape=(2,4,9,9))
hist = [0 for _ in range(8)]
click = 0


def mouse_event(event, x, y, flags, param):
    global click, hoS, hoG, patch
    patch_size = 4
    if event == cv2.EVENT_FLAG_LBUTTON:
        print("x =", x, ", y = ", y, ", click = ", click)
        patch[click // 4 % 2][click % 4] = param[y-patch_size-1:y+patch_size, x-patch_size-1:x+patch_size]

        # numpy.array(list,type) <-- 여기서 마이너스 값을 얻기위해 이미지를 integer로 바꿔주었다
        # x 축 변화량 = (col+1) - (col-1)
        xgrad = np.array(param[y-patch_size-1:y+patch_size, x-patch_size:x+patch_size+1], int) \
                - np.array(param[y-patch_size-1:y+patch_size, x-patch_size-2:x+patch_size-1], int)

        # y 축 변화량 = (row-1) - (row+1)
        ygrad = np.array(param[y - patch_size - 2:y + patch_size - 1, x - patch_size - 1:x + patch_size], int) \
                - np.array(param[y - patch_size:y + patch_size + 1, x - patch_size - 1:x + patch_size], int)

        # print(param[y+patch_size][x+patch_size+1])
        # print(param[y+patch_size][x+patch_size-1])
        # print(np.array(param[y+patch_size][x+patch_size+1],int)-np.array(param[y+patch_size][x+patch_size-1],int))
        # print("xgrad")
        # print(xgrad)
        # print("ygrad")
        # print(ygrad)
        # 크기 = 제곱 합 에 루트 => 9*9
        grad_size = xgrad**2 + ygrad**2
        hoS[click // 4 % 2][click % 4] = np.sqrt(grad_size)

        # 방향 = 아크탄젠트
        grad_direction = np.arctan2(xgrad, ygrad)
        grad_direction = grad_direction * 180 / np.pi
        hoG[click // 4 % 2][click % 4] = (grad_direction + 360) % 360
        print("grad_size")
        print(grad_size)
        print("grad_direction")
        print(hoG[click // 4 % 2][click % 4])
        # gradient histogram 저장해서, 한번에 보여줘야한다.
        # plt.hist(np.array(hoG[click // 4 % 2][click % 4]).flatten(), bins=8, label="grad histo")
        # plt.legend()
        # plt.show()


        # for i in range(9):
        #     for j in range(9):
        #         print(patch[click // 4 % 2][click % 4][i][j], end=" ")
        #     print()
        # print("----------------------------")
        cv2.putText(param, str(click % 4), (x, y),fontFace=cv2.FONT_ITALIC, fontScale=1.5, thickness=1,color=(250,250,250), lineType=cv2.LINE_AA )
        cv2.rectangle(param, (x-patch_size, y-patch_size), (x+patch_size, y+patch_size), 1)
        click += 1


src = np.full((500, 500), 255, dtype=np.uint8)
src1 = cv2.imread("1st.jpg", cv2.IMREAD_GRAYSCALE)
src2 = cv2.imread("2nd.jpg", cv2.IMREAD_GRAYSCALE)

print(np.array([[1, 0, 2], [2, 3, 3], [1, 2, 0]]) - np.array([[3, 2, 1], [0, 1, 2], [3, 0, 1]]))
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
dst1 = cv2.resize(src1, dsize=(0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
dst2 = cv2.resize(src2, dsize=(0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
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

# cv2.imshow("e",src)
# cv2.setMouseCallback("e",mouse_event,src)
# cv2.waitKey()
cv2.destroyAllWindows()