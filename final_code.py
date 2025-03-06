import cv2
import numpy as np
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Set properties for volume and speech rate
# engine.setProperty('volume', 1.0)  # Set to maximum volume
# engine.setProperty('rate', 150)     # Moderate speech rate

# List available voices and choose a female voice
voices = engine.getProperty('voices')
# print("Available voices:")
# for voice in voices:
#     print(f"Name: {voice.name}, ID: {voice.id}")

female_voice = None
for voice in voices:
    if 'female' in voice.name.lower() or 'zira' in voice.id.lower():  # Check for female voice
        # engine.setProperty('voice', voice.id)
        # print(f"Selected voice: {voice.name}")
        female_voice =voice.id
        break


# Load YOLO model
net = cv2.dnn.readNet(
    r"C:\\Users\\\Srushti Patange\\OneDrive\\Desktop\\Projects\\final year project\\final_year_Project_code_\\yolov3 (1).cfg",
    r"C:\\Users\\Srushti Patange\\OneDrive\Desktop\\Projects\\final year project\final_year_Project_code_\\yolov3 (1).weights")


layer_names = net.getLayerNames()

# Get output layer names
unconnected_layers = net.getUnconnectedOutLayers()
if len(unconnected_layers.shape) == 2:
    output_layers = [layer_names[i[0] - 1] for i in unconnected_layers]
else:
    output_layers = [layer_names[i - 1] for i in unconnected_layers]

# Load COCO object classes
with open('C:\\Users\\Srushti Patange\\OneDrive\\Desktop\\Projects\\final year project\\final_year_Project_code_\\coco (1).names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set to store already detected objects
detected_objects = set()

# Set detection thresholds
confidence_threshold = 0.6  # Increase confidence threshold
nms_threshold = 0.4          # Adjust NMS threshold

def speak(text):
    """Function to convert text to speech."""
    print(f"Speaking: {text}")  # Debug statement
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"An error occurred during speech synthesis: {e}")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    height, width, channels = frame.shape

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False
    )
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists to store detection data
    class_ids = []
    confidences = []
    boxes = []

    # Analyze detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:  # Use increased threshold
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Get rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                label = str(classes[class_id])
                print(f"Detected {label} with confidence {confidence}")

    # Apply Non-maxima Suppression (NMS) with adjusted threshold
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    # Flatten the indexes list
    if len(indexes) > 0:
        indexes = indexes.flatten()
    else:
        indexes = []

    multiple_labels = []  # List to accumulate labels for output

    for i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])

        if label not in detected_objects:
            detected_objects.add(label)
            multiple_labels.append(label)  # Append label for this object

    # Speak all the newly detected labels at once
    if multiple_labels:
        print(f"Multiple labels detected: {multiple_labels}")  # Debug statement
        speak("I see " + ", ".join(multiple_labels))  # Announce multiple detected objects

    # Draw bounding boxes and labels for all detected objects
    for label in detected_objects:
        # Get the last known position of the detected label from boxes
        # This step assumes that the detected_objects set maintains the order of detection
        for i in range(len(boxes)):
            if class_ids[i] == classes.index(label):  # Check if the class_id matches
                x, y, w, h = boxes[i]
                color = (0, 255, 0)  # Green color for the bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
                break  # Exit the loop after drawing the label

    # Save the frame with detections
    cv2.imwrite('output.jpg', frame)

    # Display the resulting frame in real-time
    cv2.imshow('Frame', frame)

    # Exit the program when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
