
# # demo15_load_model_live_preview
# import cv2
# import torchvision.transforms as transforms
# import torchvision
# import torch
# import PIL
#
# MODEL_WEIGHT = 'model/weight_only'
# # for first camera, use 0
# cap = cv2.VideoCapture(0)
#
# device = torch.device('cpu')
# model1 = torchvision.models.resnet18()  # CNN Model
# print('model1 = ',model1)
# model1 = model1.to(device)
# model1.fc = torch.nn.Linear(512, 2)
# model1.load_state_dict(torch.load(MODEL_WEIGHT))
#
# TRANSFORMS = transforms.Compose([
#     transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),   # 亮度
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # mean & std
# ])
#
# def verify(f):
#     image = f.to(device)
#     image = torch.reshape(image, [1, 3, 224, 224])  # CNN for 4D
#     output = model1(image)
#     if output.argmax(1) == 1:
#         print('down')
#     else:
#         print("up")
#
# while True:
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('frame', gray)
#     inputKey = cv2.waitKey(1) & 0xFF  # cv2.waitKey(1): 取按键的ASCII值，0xFF 返回值最后八位
#     if inputKey == ord('q'):
#         break
#     elif inputKey == ord('v'):
#         print("call pytorch")
#         frame = PIL.Image.fromarray(frame) # transfor tensor
#         frame = TRANSFORMS(frame)
#         verify(frame)
#
# cap.release()
# cv2.destroyAllWindows()


