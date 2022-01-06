# Demo on csgo auto-aim

main libraries:

1. pytorch - yolov5s6
   1. to find the person in the screenshots

2. win32api 
   1. to control the keyboard and mouse
   2. to grab the screenshots

just run the `csgo_yolov5s6.ipynb` file to see the results.

`csgo_yolofastestv2.py` is another implementation but yolo-fastest v2 has bad classification accuracy.

for more information, you can visit [ultralytics/yolov5: YOLOv5 🚀 in PyTorch > ONNX > CoreML > TFLite (github.com)](https://github.com/ultralytics/yolov5) to see `yolov5` and [Load YOLOv5 from PyTorch Hub ⭐ · Issue #36 · ultralytics/yolov5 (github.com)](https://github.com/ultralytics/yolov5/issues/36) to find how to deploy yolov5 easily.