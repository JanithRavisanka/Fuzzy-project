import skfuzzy as fuzz
import skfuzzy.control as ctrl
import cv2
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

# Initialize YOLOv8 model
model = YOLO("yolov8n.pt")

# Kalman Filter Class for Object Tracking
class ObjectTracker:
    id_counter = 0  # Class-level counter to assign unique IDs

    def __init__(self, dt=1.0, state_dim=4, meas_dim=2):
        self.kf = KalmanFilter(dim_x=state_dim, dim_z=meas_dim)

        # State transition matrix
        self.kf.F = np.array([[1, 0, dt, 0],  # x
                              [0, 1, 0, dt],  # y
                              [0, 0, 1, 0],   # vx
                              [0, 0, 0, 1]])  # vy

        # Observation model
        self.kf.H = np.array([[1, 0, 0, 0],  # x
                              [0, 1, 0, 0]])  # y

        # Covariance matrices
        self.kf.P *= 500  # Initial uncertainty
        self.kf.R = np.eye(meas_dim) * 1  # Measurement noise
        self.kf.Q = np.eye(state_dim) * 0.1  # Process noise

        # State vector
        self.kf.x = np.zeros((state_dim, 1))  # [x, y, vx, vy]

        # Assign a unique ID to this tracker
        self.id = ObjectTracker.id_counter
        ObjectTracker.id_counter += 1

        # Tracks how long the tracker has been unupdated
        self.age = 0

    def predict(self):
        self.kf.predict()
        self.age += 1
        return self.kf.x[:2].flatten()  # Predicted (x, y)

    def update(self, measurement):
        self.kf.update(measurement)
        self.age = 0


# Fuzzy Logic System for Tracker Matching
def fuzzy_iou_proximity(iou, distance):
    # Create fuzzy variables
    iou_level = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'IoU')
    distance_level = ctrl.Antecedent(np.arange(0, 200, 10), 'Distance')
    match_level = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'Match')

    # Fuzzy membership functions
    iou_level['low'] = fuzz.trimf(iou_level.universe, [0, 0, 0.5])
    iou_level['medium'] = fuzz.trimf(iou_level.universe, [0.3, 0.5, 0.7])
    iou_level['high'] = fuzz.trimf(iou_level.universe, [0.6, 1.0, 1.0])

    distance_level['near'] = fuzz.trimf(distance_level.universe, [0, 0, 100])
    distance_level['mid'] = fuzz.trimf(distance_level.universe, [50, 100, 150])
    distance_level['far'] = fuzz.trimf(distance_level.universe, [100, 200, 200])

    match_level['low'] = fuzz.trimf(match_level.universe, [0, 0, 0.5])
    match_level['medium'] = fuzz.trimf(match_level.universe, [0.3, 0.5, 0.7])
    match_level['high'] = fuzz.trimf(match_level.universe, [0.6, 1.0, 1.0])

    # Rules
    rules = [
        ctrl.Rule(iou_level['high'] & distance_level['near'], match_level['high']),
        ctrl.Rule(iou_level['medium'] & distance_level['mid'], match_level['medium']),
        ctrl.Rule(iou_level['low'] | distance_level['far'], match_level['low']),
        ctrl.Rule(iou_level['high'] & distance_level['mid'], match_level['medium']),
        ctrl.Rule(iou_level['medium'] & distance_level['near'], match_level['medium']),
        ctrl.Rule(iou_level['medium'] & distance_level['far'], match_level['low']),
        ctrl.Rule(iou_level['high'] & distance_level['far'], match_level['low']),
        ctrl.Rule(iou_level['low'] & distance_level['near'], match_level['medium']),
        ctrl.Rule(iou_level['low'] & distance_level['mid'], match_level['low'])
    ]

    # Control system
    matching_ctrl = ctrl.ControlSystem(rules)
    matching_sim = ctrl.ControlSystemSimulation(matching_ctrl)

    # Simulate
    matching_sim.input['IoU'] = iou
    matching_sim.input['Distance'] = distance
    matching_sim.compute()

    return matching_sim.output['Match']


# Helper function: Compute IoU
def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to corners
    box1 = [x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2]
    box2 = [x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2]

    # Intersection
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


# Main Tracking Function
def track_objects(video_path, max_age=10):
    cap = cv2.VideoCapture(video_path)
    trackers = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results[0].boxes.xywh.numpy()
        confs = results[0].boxes.conf.numpy()
        detections = detections[confs > 0.5]

        unmatched_trackers = []
        for det in detections:
            cx, cy, w, h = det
            measurement = np.array([[cx], [cy]])

            matched = False
            for tracker in trackers:
                pred_x, pred_y = tracker.predict()
                iou = compute_iou([pred_x, pred_y, w, h], det)
                distance = np.sqrt((cx - pred_x)**2 + (cy - pred_y)**2)

                # Fuzzy decision
                match_score = fuzzy_iou_proximity(iou, distance)
                if match_score > 0.5:  # Fuzzy threshold
                    tracker.update(measurement)
                    matched = True
                    break

            if not matched:
                new_tracker = ObjectTracker()
                new_tracker.update(measurement)
                unmatched_trackers.append(new_tracker)

        trackers = [t for t in trackers if t.age < max_age]
        trackers.extend(unmatched_trackers)

        for tracker in trackers:
            pred_x, pred_y = tracker.predict()
            x, y, w, h = pred_x, pred_y, 50, 50
            top_left = (int(x - w / 2), int(y - h / 2))
            bottom_right = (int(x + w / 2), int(y + h / 2))

            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {tracker.id}", (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Run the tracking
track_objects("vid2.mp4")