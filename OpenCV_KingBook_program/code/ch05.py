# import cv2
# import numpy as np
# height = 160  # 影像高
# width = 280  # 影像寬
# # 建立GRAY影像陣列
# image = np.zeros((height, width), np.uint8)
# cv2.imshow("image", image)  # 顯示影像
#
# image1 = np.zeros((height, width), np.uint8)
# image1.fill(255)
# cv2.imshow("image1", image1)  # 顯示影像
#
# image2 = np.ones((height, width), np.uint8) * 128
# cv2.imshow("image2", image2)  # 顯示影像
#
# image3 = np.zeros((height, width), np.uint8)
# image3[40:120, 70:210] = 255 # 高在40至120之間,寬在70至210之間,設為255
# cv2.imshow("image3[40:120,70:210]=255", image3)  # 顯示影像
#
# image4 = np.zeros((height, width), np.uint8)
# for y in range(0, height, 20):
#     image4[y:y+10, :] = 255  # 白色厚度是10
# cv2.imshow("image4", image4)  # 顯示影像
#
# # 使用random.randint()建立GRAY影像陣列
# image5 = np.random.randint(256,size=[height, width],dtype=np.uint8)
# cv2.imshow("randint(256)image5", image5)  # 顯示影像
#
# # 建立BGR影像陣列
# image6 = np.zeros((height, width, 3), np.uint8)
# image6[:,:,0] = 255              # 建立 B 通道像素值
# cv2.imshow("image6", image6)      # 顯示影像
#
# cv2.waitKey(0) ; cv2.destroyAllWindows()
#---------------------------------------------------------------------
# import cv2
# import numpy as np
#
# height = 160  # 影像高
# width = 280  # 影像寬
# # 建立BGR影像陣列
# image = np.zeros((height, width, 3), np.uint8)
# blue_image = image.copy()
# blue_image[:, :, 0] = 255  # 建立 B 通道像素值
# cv2.imshow("blue image", blue_image)  # 顯示blue image影像
#
# green_image = image.copy()
# green_image[:, :, 1] = 255  # 建立 G 通道像素值
# cv2.imshow("green image", green_image)  # 顯示green image影像
#
# red_image = image.copy()
# red_image[:, :, 2] = 255  # 建立 R 通道像素值
# cv2.imshow("red image", red_image)  # 顯示red image影像
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#----------------------------------------------------------------
# import cv2
# import numpy as np
#
# height = 160  # 影像高
# width = 280  # 影像寬
# # 使用random.randint()建立GRAY影像陣列
# image = np.random.randint(256, size=[height, width, 3], dtype=np.uint8)
# cv2.imshow("image", image)  # 顯示影像
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#-----------------------------------------------------------------
import cv2
import numpy as np

height = 150  # 影像高
width = 300  # 影像寬
image = np.zeros((height, width, 3), np.uint8)
image[0:50, :, 0] = 255  # blue
image[50:100, :, 1] = 255  # green
image[100:150, :, 2] = 255  # red
cv2.imshow("image", image)  # 顯示影像

cv2.waitKey(0)
cv2.destroyAllWindows()