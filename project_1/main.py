# 와서 할일 hog calcHist를 이용하여 구한 뒤 오차 찾기


import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from queue import PriorityQueue

que = PriorityQueue()
patch = [[[[0 for _ in range(9)] for _ in range(9)] for _ in range(4)] for _ in range(2)]
#histogram of Color
hoC = np.zeros(shape=(2,4,9,9))
#histogram of Gradient
patch_gradient = np.zeros(shape=(2,4,9,9))
hog = np.zeros(shape=(2,4,8))
#Histogram of Size
hoS = np.zeros(shape=(2,4,9,9))
#compare Histogram result
comp_hist_res = np.zeros(shape=(4,4))
hist = [0 for _ in range(8)]
# to draw final image with matching line, remember the position of mouse click
xdir = [0 for _ in range(8)]
ydir = [0 for _ in range(8)]
# to make pair of each point
pair = [-1, -1, -1, -1]
# count click
click = 0
# patch size
PatchSize = 9


def mouse_event(event, x, y, flags, param):
    global click, hoS, patch_gradient, patch, hog, PatchSize
    patch_size = PatchSize // 2
    if event == cv2.EVENT_FLAG_LBUTTON:
        # print("x =", x, ", y = ", y, ", click = ", click)
        xdir[click]=x
        ydir[click]=y
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
        hog[click // 4 % 2][click % 4], bins = np.histogram(patch_gradient[click // 4 % 2][click % 4].flatten(), np.arange(0,361,45))
        # print("grad_size")
        # print(grad_size)
        # print("grad_direction")
        # print(patch_gradient[click // 4 % 2][click % 4])
        # print("-----------calcHist-------------")
        # print(hog[click // 4 % 2][click % 4])
        cv2.putText(param, str(click % 4), (x-(patch_size*3), y-(patch_size*3)),fontFace=cv2.FONT_ITALIC, fontScale=1.5,
                    thickness=1, color=(250,250,250), lineType=cv2.LINE_AA)
        cv2.rectangle(param, (x-patch_size, y-patch_size), (x+patch_size, y+patch_size), 1)
        click += 1


def findBest():
    global comp_hist_res, hog,que,pair
    pair=[-1 for _ in range(8)]
    for i in range(4):
        for j in range(4):
            comp_hist_res[i][j] = cv2.compareHist(hog[0][i].astype('float32'), hog[1][j].astype('float32'), cv2.HISTCMP_BHATTACHARYYA)
            que.put((comp_hist_res[i][j],i,j))
    for i in range(16):
        tmp, first, second = que.get()
        if pair[first] == -1:
            pair[first] = second


def find_best_with_no_duplicate():
    global comp_hist_res, hog,que,pair
    pair=[-1 for _ in range(8)]
    visit = [-1 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            comp_hist_res[i][j] = cv2.compareHist(hog[0][i].astype('float32'), hog[1][j].astype('float32'), cv2.HISTCMP_BHATTACHARYYA)
            que.put((comp_hist_res[i][j],i,j))
    for i in range(16):
        tmp, first, second = que.get()
        if pair[first] == -1 and visit[second] == -1:
            pair[first] = second
            visit[second] = 1


def plot_histogram_of_gradient():
    degree = ['0', '45', '90', '135', '180', '225', '270', '315']
    plt.figure(figsize=(16, 8))
    plt.subplot(241).set_title("1-0")
    plt.bar(np.arange(8), hog[0][0], align='edge', tick_label=degree)

    plt.subplot(242).set_title("1-1")
    plt.bar(np.arange(8), hog[0][1], align='edge', tick_label=degree)

    plt.subplot(243).set_title("1-2")
    plt.bar(np.arange(8), hog[0][2], align='edge', tick_label=degree)

    plt.subplot(244).set_title("1-3")
    plt.bar(np.arange(8), hog[0][3], align='edge', tick_label=degree)

    plt.subplot(245).set_title("2-0")
    plt.bar(np.arange(8), hog[1][0], align='edge', tick_label=degree)

    plt.subplot(246).set_title("2-1")
    plt.bar(np.arange(8), hog[1][1], align='edge', tick_label=degree)

    plt.subplot(247).set_title("2-2")
    plt.bar(np.arange(8), hog[1][2], align='edge', tick_label=degree)

    plt.subplot(248).set_title("2-3")
    plt.bar(np.arange(8), hog[1][3], align='edge', tick_label=degree)
    plt.show()


src2 = cv2.imread("2nd.jpg", cv2.IMREAD_GRAYSCALE)
src1 = cv2.imread("1st.jpg", cv2.IMREAD_GRAYSCALE)
# resize the images
scaling = 0.1
dst1 = cv2.resize(src1, dsize=(0, 0), fx=scaling, fy=scaling, interpolation=cv2.INTER_AREA)
dst2 = cv2.resize(src2, dsize=(0, 0), fx=scaling, fy=scaling, interpolation=cv2.INTER_AREA)
dst1 = cv2.GaussianBlur(dst1,(5,5),0)
dst2 = cv2.GaussianBlur(dst2,(5,5),0)
addh = cv2.hconcat([dst1, dst2])
height, width = dst1.shape

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

plot_histogram_of_gradient()
findBest()
# find_best_with_no_duplicate()

# while(True):
#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:
#         break

# draw rectangle and text on Adding image
for i in range(8):
    if i <=3:
        cv2.putText(addh, str(i), (xdir[i]-((PatchSize//2)*3), ydir[i]-((PatchSize//2)*3)),fontFace=cv2.FONT_ITALIC, fontScale=1.5,
                    thickness=1, color=(250,250,250), lineType=cv2.LINE_AA)
        cv2.rectangle(addh, (xdir[i]-(PatchSize//2), ydir[i]-(PatchSize//2)), (xdir[i]+(PatchSize//2), ydir[i]+(PatchSize//2)), 1)
    else:
        cv2.putText(addh, str(i%4), (width +xdir[i]-((PatchSize//2)*3), ydir[i]-((PatchSize//2)*3)),fontFace=cv2.FONT_ITALIC, fontScale=1.5,
                    thickness=1, color=(250,250,250), lineType=cv2.LINE_AA)
        cv2.rectangle(addh, (width + xdir[i]-(PatchSize//2), ydir[i]-(PatchSize//2)), (width + xdir[i]+(PatchSize//2), ydir[i]+(PatchSize//2)), 1)

# draw pair line
for i in range(4):
    cv2.line(addh, (xdir[i], ydir[i]), (width + xdir[pair[i]+4], ydir[pair[i]+4]), (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

cv2.imshow("addh", addh)
cv2.waitKey()
cv2.destroyAllWindows()