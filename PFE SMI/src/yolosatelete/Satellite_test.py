from ultralytics import YOLO
#==->>entrainement
#model=YOLO("yolov8n.pt")
#model.train(data="data.yaml",epochs=100,imgsz=640)
#print("entrainement succès")

#==->>detectio
model=YOLO("runs/detect/train/weights/best.pt")
results=model("outimg/OBJ04943_PS3_K3A_NIA0326_png.rf.f71dee8a7ae5c809cd4b5e61569cb992.jpg",conf=0.2)
results[0].show()

