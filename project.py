from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.yaml')

# Train
results = model.train(data='config.yaml', epochs=3, project='results')
  # save, view, and plot training results
