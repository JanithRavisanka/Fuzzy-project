import cv2
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import concurrent.futures

# Initialize YOLOv8 model
model = YOLO("yolo11l.pt")

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
        self.kf.P *= 1000  # Initial uncertainty
        self.kf.R = np.eye(meas_dim) * 1  # Measurement noise
        self.kf.Q = np.eye(state_dim) * 0.1  # Process noise

        # State vector
        self.kf.x = np.zeros((state_dim, 1))  # [x, y, vx, vy]

        # Assign a unique ID to this tracker
        self.id = ObjectTracker.id_counter
        ObjectTracker.id_counter += 1

        # Tracks how long the tracker has been unupdated
        self.age = 0
        self.color = None  # Store the object's color

    def predict(self):
        self.kf.predict()
        self.age += 1
        return self.kf.x[:2].flatten()  # Predicted (x, y)

    def update(self, measurement, color=None):
        self.kf.update(measurement)
        self.age = 0
        if color is not None:
            self.color = color  # Update color


# Fuzzy Logic System for Tracker Matching
def fuzzy_iou_proximity(iou, distance, color_similarity):
    # Create fuzzy variables
    iou_level = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'IoU')
    distance_level = ctrl.Antecedent(np.arange(0, 200, 10), 'Distance')
    color_level = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'ColorSimilarity')
    match_level = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'Match')

    # Simplified membership functions
    iou_level['low'] = fuzz.trapmf(iou_level.universe, [0, 0, 0.3, 0.5])
    iou_level['high'] = fuzz.trapmf(iou_level.universe, [0.3, 0.5, 1.0, 1.0])

    distance_level['near'] = fuzz.trapmf(distance_level.universe, [0, 0, 50, 100])
    distance_level['far'] = fuzz.trapmf(distance_level.universe, [50, 100, 200, 200])

    color_level['low'] = fuzz.trapmf(color_level.universe, [0, 0, 0.3, 0.5])
    color_level['high'] = fuzz.trapmf(color_level.universe, [0.3, 0.5, 1.0, 1.0])

    match_level['low'] = fuzz.trapmf(match_level.universe, [0, 0, 0.3, 0.5])
    match_level['high'] = fuzz.trapmf(match_level.universe, [0.3, 0.5, 1.0, 1.0])

    # Simplified rules
    rules = [
        # High match conditions
        ctrl.Rule(
            iou_level['high'] & distance_level['near'] & color_level['high'],
            match_level['high']
        ),
        
        # Medium match conditions
        ctrl.Rule(
            (iou_level['high'] & distance_level['far']) |
            (iou_level['low'] & distance_level['near'] & color_level['high']),
            match_level['high']
        ),
        
        # Low match conditions
        ctrl.Rule(
            (iou_level['low'] & distance_level['far']) |
            (iou_level['low'] & color_level['low']),
            match_level['low']
        )
    ]

    # Control system
    matching_ctrl = ctrl.ControlSystem(rules)
    matching_sim = ctrl.ControlSystemSimulation(matching_ctrl)

    # Simulate
    try:
        matching_sim.input['IoU'] = iou
        matching_sim.input['Distance'] = distance
        matching_sim.input['ColorSimilarity'] = color_similarity
        matching_sim.compute()
        return matching_sim.output['Match']
    except Exception as e:
        print(f"Error in fuzzy system: {e}. Returning default match score of 0.")
        return 0


# Helper Functions
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


# Get the dominant color in a bounding box by averaging pixel colors
def get_dominant_color(frame, bbox):
    x, y, w, h = bbox
    x1, y1 = max(0, int(x - w / 2)), max(0, int(y - h / 2))
    x2, y2 = min(frame.shape[1], int(x + w / 2)), min(frame.shape[0], int(y + h / 2))

    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        return np.array([0, 0, 0])  # Default to black if the crop is invalid

    # Compute mean color
    mean_color = np.mean(cropped, axis=(0, 1))  # Average over height and width
    return mean_color  # Returns [B, G, R]


# Compute color similarity between two colors by Euclidean distance
def compute_color_similarity(color1, color2):
    if color1 is None or color2 is None:
        return 0  # No similarity if either color is missing
    return 1 - (np.linalg.norm(color1 - color2) / np.sqrt(255**2 * 3))  # Normalize to [0, 1]


# Optimized Main Tracking Function with Better Bounding Boxes
def track_objects(video_path, max_age=10):
    cap = cv2.VideoCapture(video_path)
    trackers = []

    def process_frame(frame):
        results = model(frame)
        detections = results[0].boxes.xywh.numpy()
        confs = results[0].boxes.conf.numpy()
        detections = detections[confs > 0.5]

        unmatched_trackers = []
        # Store original dimensions with trackers
        original_dims = {}  # Dictionary to store width and height for each tracker

        for det in detections:
            cx, cy, w, h = det
            measurement = np.array([[cx], [cy]])
            color = get_dominant_color(frame, [cx, cy, w, h])

            matched = False
            for tracker in trackers:
                pred_x, pred_y = tracker.predict()
                iou = compute_iou([pred_x, pred_y, w, h], det)
                distance = np.sqrt((cx - pred_x)**2 + (cy - pred_y)**2)
                color_similarity = compute_color_similarity(tracker.color, color)

                # Fuzzy decision
                match_score = fuzzy_iou_proximity(iou, distance, color_similarity)
                if match_score > 0.5:  # Fuzzy threshold
                    tracker.update(measurement, color)
                    original_dims[tracker.id] = (w, h)  # Store dimensions
                    matched = True
                    break

            if not matched:
                new_tracker = ObjectTracker()
                new_tracker.update(measurement, color)
                original_dims[new_tracker.id] = (w, h)  # Store dimensions for new tracker
                unmatched_trackers.append(new_tracker)

        trackers[:] = [t for t in trackers if t.age < max_age]
        trackers.extend(unmatched_trackers)

        # Draw bounding boxes using original dimensions
        for tracker in trackers:
            pred_x, pred_y = tracker.predict()
            if tracker.id in original_dims:
                w, h = original_dims[tracker.id]
                top_left = (int(pred_x - w/2), int(pred_y - h/2))
                bottom_right = (int(pred_x + w/2), int(pred_y + h/2))
                
                # Draw bounding box
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                
                # Draw ID with background for better visibility
                label = f"ID {tracker.id}"
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                # Draw background rectangle for text
                cv2.rectangle(
                    frame,
                    (top_left[0], top_left[1] - label_height - 10),
                    (top_left[0] + label_width, top_left[1]),
                    (0, 0, 0),
                    -1,
                )
                
                # Draw ID text
                cv2.putText(
                    frame,
                    label,
                    (top_left[0], top_left[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

        return frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame)
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Test the tracking with a video
track_objects("vid4.mkv")
