import cv2
import numpy as np
import argparse
import win32api,win32gui,win32ui,win32con
import torch
import time
import pyautogui
import pydirectinput


class yolo_fast_v2():
    def __init__(self, objThreshold=0.3, confThreshold=0.3, nmsThreshold=0.4):
        with open('coco.names', 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')   ###这个是在coco数据集上训练的模型做opencv部署的，如果你在自己的数据集上训练出的模型做opencv部署，那么需要修改self.classes
        self.stride = [16, 32]
        self.anchor_num = 3
        self.anchors = np.array([12.64, 19.39, 37.88, 51.48, 55.71, 138.31, 126.91, 78.23, 131.57, 214.55, 279.92, 258.87],
                           dtype=np.float32).reshape(len(self.stride), self.anchor_num, 2)
        self.inpWidth = 352
        self.inpHeight = 352
        self.net = cv2.dnn.readNet('model.onnx')
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold
    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def postprocess(self, frame, outs):
        # 原始
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        ratioh, ratiow = frameHeight / self.inpHeight, frameWidth / self.inpWidth
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        lis = []
        for detection in outs:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > self.confThreshold and detection[4] > self.objThreshold and classId == 0:
                center_x = int(detection[0] * ratiow)
                center_y = int(detection[1] * ratioh)
                width = int(detection[2] * ratiow)
                height = int(detection[3] * ratioh)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                # confidences.append(float(confidence))
                confidences.append(float(confidence*detection[4]))
                boxes.append([left, top, width, height])
                lis.append([center_x,center_y])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        # 获取去重后的box索引
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        # 存储box的left,top,width,height
        lis2 = []
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            center_x = lis[i][0]
            center_y = lis[i][1]
            lis2.append([left,top,width,height,center_x,center_y,(abs(center_x-208)+abs(center_y-208))])
        #     frame = self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
        # return frame
        if len(lis2) == 0:
            return
        else:
            lis2 = sorted(lis2,key = lambda x:x[6])
            return lis2[0]

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)

        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=1)
        return frame
    def detect(self, srcimg):
        blob = cv2.dnn.blobFromImage(srcimg, 1 / 255.0, (self.inpWidth, self.inpHeight))
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())[0]

        outputs = np.zeros((outs.shape[0]*self.anchor_num, 5+len(self.classes)))
        row_ind = 0
        for i in range(len(self.stride)):
            h, w = int(self.inpHeight / self.stride[i]), int(self.inpWidth / self.stride[i])
            length = int(h * w)
            grid = self._make_grid(w, h)
            for j in range(self.anchor_num):
                top = row_ind+j*length
                left = 4*j
                outputs[top:top + length, 0:2] = (outs[row_ind:row_ind + length, left:left+2] * 2. - 0.5 + grid) * int(self.stride[i])
                outputs[top:top + length, 2:4] = (outs[row_ind:row_ind + length, left+2:left+4] * 2) ** 2 * np.repeat(self.anchors[i, j, :].reshape(1,-1), h * w, axis=0)
                outputs[top:top + length, 4] = outs[row_ind:row_ind + length, 4*self.anchor_num+j]
                outputs[top:top + length, 5:] = outs[row_ind:row_ind + length, 5*self.anchor_num:]
            row_ind += length
        return outputs
CUDA = torch.cuda.is_available() 
num_classes = 80  # coco 数据集有80类
objThreshold = 0.2
confThreshold = 0.3
nmsThreshold = 0.3 #Non-maximum suppression threshold
print("loading model...")
model = yolo_fast_v2(objThreshold, confThreshold, nmsThreshold)
# model.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
# model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
print("success!")
# t1 = time.time()
# srcimg = cv2.imread(r'C:\Users\JPY\Desktop\csgo_aim\img\Snipaste_2022-01-04_22-42-23.png')
# outputs = model.detect(srcimg)
# print(outputs.shape)
# srcimg = model.postprocess(srcimg, outputs)
# print(srcimg.shape)
# t2 = time.time()
# print(t2-t1)
# winName = 'Deep learning object detection in OpenCV'
# cv2.namedWindow(winName, 0)
# cv2.imshow(winName, srcimg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
while True:
    MouseX, MouseY = pyautogui.position()
    two_aims = []
    for i in range(2):
        # 截图
        hwnd = 0  # 窗口的编号，0号表示当前活跃窗口
        # 根据窗口句柄获取窗口的设备上下文DC（Divice Context）
        hwndDC = win32gui.GetWindowDC(hwnd)
        # 根据窗口的DC获取mfcDC
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        # mfcDC创建可兼容的DC
        saveDC = mfcDC.CreateCompatibleDC()
        # 创建bigmap准备保存图片
        saveBitMap = win32ui.CreateBitmap()
        # 获取监控器信息
        MoniterDev = win32api.EnumDisplayMonitors(None, None)
        # w = MoniterDev[0][2][2]
        # h = MoniterDev[0][2][3]
        # 为bitmap开辟空间
        # saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
        saveBitMap.CreateCompatibleBitmap(mfcDC, 416, 416)
        # 高度saveDC，将截图保存到saveBitmap中
        saveDC.SelectObject(saveBitMap)
        # 截取从左上角（0，0）长宽为（w，h）的图片
        # saveDC.BitBlt((0, 0), (w, h), mfcDC, (0, 0), win32con.SRCCOPY)
        saveDC.BitBlt((0,0), (416, 416), mfcDC, (1072, 592), win32con.SRCCOPY)
        # saveBitMap.SaveBitmapFile(saveDC, 'filename')
        signedIntsArray = saveBitMap.GetBitmapBits(True)
        im_opencv = np.frombuffer(signedIntsArray, dtype = 'uint8')
        im_opencv.shape = (416, 416, 4)
        srcimg = cv2.cvtColor(im_opencv, cv2.COLOR_BGRA2RGB)
        outputs = model.detect(srcimg)
        # 存储距离中心点最近的box left,top,width,height,center_x,center_y,distance
        box= model.postprocess(srcimg, outputs)
        if not box:
            continue
        aim_x = int(0.5*(box[0]) + 0.5*(box[0]+box[2]))
        aim_x -= 208
        aim_y = int(0.6*(box[1]) + 0.4*(box[1] + box[3]))
        aim_y -= 208
        two_aims.append([aim_x,aim_y])

    if len(two_aims) < 2:
        continue
    else:
        aimx_1,aimy_1 = two_aims[0]
        aimx_2,aimy_2 = two_aims[1]
        distance = abs(aimx_1-aimx_2) + abs(aimy_1-aimy_2)
        if distance > 50:
            continue
        else:
            delta_x = aimx_2 - aimx_1
            delta_y = aimy_2 - aimy_1
            aim_x = aimx_2 + 6*delta_x
            aim_y = aimy_2 + 3*delta_y
    
    pydirectinput.moveTo(int(aim_x/1.5)+1280, int(aim_y/1.5)+800,duration=0)
    pydirectinput.click()
    # pydirectinput.click()
    #内存释放
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd,hwndDC)
    
    


