import numpy as np
import cv2 as cv
from keras.models import load_model

def main():
    # Read yolo config
    net = cv.dnn.readNet("models/yolov3_training_final.weights", "models/yolov3_testing.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[idx - 1] for idx in net.getUnconnectedOutLayers()]

    # Set CUDA if available
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

    # Load custom CNN model
    model = load_model("models/TSR_model.h5")
    threshold = 0.5
    font = cv.FONT_HERSHEY_COMPLEX_SMALL

    # Helpers for video input
    video = cv.VideoCapture("input/rome.mp4")

    # Read classes
    classes4detection = []
    with open("classes/classes_detection.names", "r") as classes_file:
        classes4detection = [class_name.strip() for class_name in classes_file.readlines()]

    classes4recognition = []
    with open("classes/classes_recognition.names", "r") as classes_file:
        classes4recognition = [class_name.strip() for class_name in classes_file.readlines()]
        roi_width = 30
        roi_height = 30
        color = (227, 255, 0) # green lime
        images = []
        # Insert input video into pipeline
        while True:
            status, image = video.read()
            if not status:
                break
            try:
                height, width, _ = image.shape

                # Detection
                image_blob = cv.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
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
                        image = cv.rectangle(image, (x, y), (x + w, y + h),  color, 2)
                        # Crop to ROI
                        crop_image = image[y: y + h, x: x + w]
                        if len(crop_image) > 0:
                            crop_image = cv.resize(crop_image, (roi_width, roi_height))
                            crop_image = cv.cvtColor(crop_image, cv.COLOR_BGR2GRAY)
                            crop_image = crop_image.reshape(-1, roi_width, roi_height, 1)
                            # Make prediction
                            prediction = np.argmax(model.predict(crop_image))
                            # Label for recognition classes + confidence percentage
                            label = str(classes4recognition[prediction]) + "=" + str(round(model.predict(crop_image)[0][prediction] * 100, 2)) + "%"
                            image = cv.putText(image, label, (x, y), font, 1,  color, 1)
                images.append(image)
                # Display result
                cv.imshow("Image", image)
                if cv.waitKey(1) & 0xFF == ord ('q'):
                    break
            except:
                pass
        
        # Save previewed video
        height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
        width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
        output_video = cv.VideoWriter('demo/output_video.avi',cv.VideoWriter_fourcc(*'DIVX'), 30, (width, height))
        for idx in range(len(images)):
            output_video.write(images[idx])
        output_video.release()

        video.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()