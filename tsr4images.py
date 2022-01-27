import numpy as np
import cv2 as cv
import os
from keras.models import load_model

def main():
    # Read input images
    images_path = "GTSDB"
    images = []
    for image_name in os.listdir(images_path):
        try:
            image = cv.imread(os.path.join(images_path, image_name))
            images.append(image)
        except:
            raise ValueError("Error loading image!")

    # Read yolo config
    net = cv.dnn.readNet("models/yolov3_training_final.weights", "models/yolov3_testing.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[idx - 1] for idx in net.getUnconnectedOutLayers()]

    # Load custom CNN model
    model = load_model("models/TSR_model.h5")
    threshold = 0.5
    font = cv.FONT_HERSHEY_TRIPLEX

    # Read classes
    classes4detection = []
    with open("classes/classes_detection.names", "r") as classes_file:
        classes4detection = [class_name.strip() for class_name in classes_file.readlines()]
    colors = np.random.uniform(0, 255, size=(len(classes4detection), 3))

    classes4recognition = []
    with open("classes/classes_recognition.names", "r") as classes_file:
        classes4recognition = [class_name.strip() for class_name in classes_file.readlines()]
        roi_width = 30
        roi_height = 30
        # Insert input image into pipeline
        for idx in range(len(images)):
            height, width, _ = images[idx].shape

            # Detection
            image_blob = cv.dnn.blobFromImage(images[idx], 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(image_blob)
            outs = net.forward(output_layers)

            # Display info
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > threshold:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Bounding box coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            # Recognition
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    # Label for detection classes + confidence percentage
                    label = str(classes4detection[class_ids[i]]) + "=" + str(round(confidences[i] * 100, 2)) + "%"
                    images[idx] = cv.rectangle(images[idx], (x, y), (x + w, y + h),  colors[class_ids[i]], 2)
                    # Crop to ROI
                    crop_image = images[idx][y: y + h, x: x + w]
                    if len(crop_image) > 0:
                        crop_image = cv.resize(crop_image, (roi_width, roi_height))
                        crop_image = cv.cvtColor(crop_image, cv.COLOR_BGR2GRAY)
                        crop_image = crop_image.reshape(-1, roi_width, roi_height, 1)
                        # Make prediction
                        prediction = np.argmax(model.predict(crop_image))
                        print(prediction)
                        # Label for recognition classes + confidence percentage
                        label = str(classes4recognition[prediction]) + "=" + str(round(model.predict(crop_image)[0][prediction] * 100, 2)) + "%"
                        images[idx] = cv.putText(images[idx], label, (x, y), font, 1,  colors[class_ids[i]], 2)
            # Display result
            cv.imshow("Image", images[idx])
            cv.waitKey(0)
            
        cv.destroyAllWindows()
if __name__ == "__main__":
    main()