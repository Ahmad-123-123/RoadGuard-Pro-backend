# Save this as test.py in your backend folder
from ultralytics import YOLO
model = YOLO("best.pt")
results = model.predict(source="test_pothole.jpg", conf=0.25, save=True)
for r in results:
    print(f"Boxes: {len(r.boxes)}")
    for b in r.boxes:
        print(f"  conf={float(b.conf[0]):.2f}")