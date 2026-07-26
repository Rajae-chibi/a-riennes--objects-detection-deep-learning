from ultralytics import YOLO
#==->>entrainement
#model=YOLO("yolov8n.pt")
#model.train(data="data.yaml",epochs=100,imgsz=640)
#print("entrainement succès")

#==->>detection
model=YOLO("runs/detect/train/weights/best.pt")
results=model("imgout/30098_jpg.rf.8e61ae1b6d7c6e1114ee9b26e80f5841.jpg",conf=0.3)
results[0].show()

