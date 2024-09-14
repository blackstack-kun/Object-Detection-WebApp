import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, send_file
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the object detection model
model = tf.saved_model.load('./ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model')

# Load class names
class_names = {}
current_id = None
with open('./mscoco_complete_label_map.pbtxt', 'r') as f:
    for line in f:
        if "id:" in line:
            current_id = int(line.strip().split(' ')[-1])
        if "display_name:" in line:
            display_name = line.strip().split('"')[1]
            class_names[current_id] = display_name

# Function to draw boxes
def draw_boxes(frame, boxes, classes, scores, min_score_thresh=.5):
    for i in range(boxes.shape[1]):
        if scores[0, i] > min_score_thresh:
            box = boxes[0, i] * np.array([frame.shape[0], frame.shape[1], frame.shape[0], frame.shape[1]])
            class_id = classes[0, i]
            class_name = class_names.get(class_id, 'N/A')
            cv2.rectangle(frame, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (int(box[1]), int(box[0] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Image upload directory
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check if the file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for uploading image
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            result_image_path = detect_objects(file_path)
            return send_file(result_image_path, mimetype='image/jpeg')

    return render_template('upload.html')

# Function to perform object detection on the uploaded image
def detect_objects(image_path):
    frame = cv2.imread(image_path)

    # Object detection
    input_tensor = tf.convert_to_tensor([frame], dtype=tf.uint8)
    detections = model(input_tensor)

    # Extracting detection results
    boxes = detections['detection_boxes'].numpy()
    classes = detections['detection_classes'].numpy().astype(np.int32)
    scores = detections['detection_scores'].numpy()

    # Draw boxes on the image
    draw_boxes(frame, boxes, classes, scores)

    # Save the processed image
    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + os.path.basename(image_path))
    cv2.imwrite(output_image_path, frame)

    return output_image_path

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
