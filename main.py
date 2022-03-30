# 와서 할일 hog calcHist를 이용하여 구한 뒤 오차 찾기


import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

patch = [[[[0 for _ in range(9)] for _ in range(9)] for _ in range(4)] for _ in range(2)]
#histogram of Color
hoC = np.zeros(shape=(2,4,9,9))
#histogram of Gradient
patch_gradient = np.zeros(shape=(2,4,9,9))
hog = np.zeros(shape=(2,4,8))
#Histogram of Size
hoS = np.zeros(shape=(2,4,9,9))
hist = [0 for _ in range(8)]
click = 0


def mouse_event(event, x, y, flags, param):
    global click, hoS, patch_gradient, patch, hog
    patch_size = 4
    if event == cv2.EVENT_FLAG_LBUTTON:
        print("x =", x, ", y = ", y, ", click = ", click)
        # 패치 global 저장
        patch[click // 4 % 2][click % 4] = param[y-patch_size-1:y+patch_size, x-patch_size-1:x+patch_size]

        # numpy.array(list,type) <-- 여기서 마이너스 값을 얻기위해 이미지를 integer로 바꿔주었다
        # x 축 변화량 = (col+1) - (col-1)
        xgrad = np.array(param[y-patch_size-1:y+patch_size, x-patch_size:x+patch_size+1], int) \
                - np.array(param[y-patch_size-1:y+patch_size, x-patch_size-2:x+patch_size-1], int)

        # y 축 변화량 = (row-1) - (row+1)
        ygrad = np.array(param[y - patch_size - 2:y + patch_size - 1, x - patch_size - 1:x + patch_size], int) \
                - np.array(param[y - patch_size:y + patch_size + 1, x - patch_size - 1:x + patch_size], int)

        # 크기 = 제곱 합 에 루트 => 9*9
        grad_size = xgrad**2 + ygrad**2
        # save the Size global
        hoS[click // 4 % 2][click % 4] = np.sqrt(grad_size)

        # 방향 = {아크탄젠트(x변화량, y변화량) * 180 / pi} --> -180 ~ 180 도 까지 나옴
        grad_direction = np.arctan2(xgrad, ygrad)
        grad_direction = grad_direction * 180 / np.pi

        # 각도를 모두 양수로 하기 위해 : (각도 + 360) % 360
        # save the Degree global
        patch_gradient[click // 4 % 2][click % 4] = (grad_direction + 360) % 360
        print("grad_size")
        print(grad_size)
        print("grad_direction")
        print(patch_gradient[click // 4 % 2][click % 4])
        # gradient histogram 저장해서, 한번에 보여줘야한다.
        # plt.hist(np.array(patch_gradient[click // 4 % 2][click % 4]).flatten(), bins=8, label="grad histo")
        # plt.legend()
        # plt.show()
        cv2.putText(param, str(click % 4), (x-(patch_size*3), y-(patch_size*3)),fontFace=cv2.FONT_ITALIC, fontScale=1.5, thickness=1,color=(250,250,250), lineType=cv2.LINE_AA )
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
    if click == 8:
        plt.subplot(241)
        plt.hist(np.array(patch_gradient[0][0]).flatten(), bins=8, label='1', edgecolor='whitesmoke',linewidth=1)

        plt.subplot(242)
        plt.hist(np.array(patch_gradient[0][1]).flatten(), bins=8, label='2', edgecolor='whitesmoke',linewidth=1)

        plt.subplot(243)
        plt.hist(np.array(patch_gradient[0][2]).flatten(), bins=8, label='3', edgecolor='whitesmoke',linewidth=1)

        plt.subplot(244)
        plt.hist(np.array(patch_gradient[0][3]).flatten(), bins=8, label='4', edgecolor='whitesmoke',linewidth=1)

        plt.subplot(245)
        plt.hist(np.array(patch_gradient[1][0]).flatten(), bins=8, label='5', edgecolor='whitesmoke',linewidth=1)

        plt.subplot(246)
        plt.hist(np.array(patch_gradient[1][1]).flatten(), bins=8, label='6', edgecolor='whitesmoke',linewidth=1)

        plt.subplot(247)
        plt.hist(np.array(patch_gradient[1][2]).flatten(), bins=8, label='7', edgecolor='whitesmoke',linewidth=1)

        plt.subplot(248)
        plt.hist(np.array(patch_gradient[1][3]).flatten(), bins=8, label='8', edgecolor='whitesmoke',linewidth=1)
        plt.show()
        click=0

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break




# cv2.imshow("e",src)
# cv2.setMouseCallback("e",mouse_event,src)
# cv2.waitKey()
cv2.destroyAllWindows()