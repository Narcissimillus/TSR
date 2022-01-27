# Traffic Sign Recogntiion (TSR) application
Traffic-Sign Recognition based on computer vision processing, deep-learning using CNN and YOLOv3.

The datasets used for training and some testing are:

* for detection: the German Trafic Sign Detection Benchmark (GTSDB);

* for recognition: the German Traffic Sign Recognition Benchmark (GTSRB) and the Belgium Traffic Sign Dataset (BTSC).

In order to run the project you will need to have Python 3 installed with all the modules included at the top of the each *.py file. You will run the project using the command 'python *.py'.

Currently the main project is divided into 2 scripts + 1 notebook, that are part of the pipeline, and were used for training/testing:

* detection.py file (with 2 additional helpers scripts: colors_select.py and helpers.py), which will return the detected traffic sign regions of an input image using the depricated method MSER, which we replaced with YOLO;

* recognition.py file, which is the script for training the custom built CNN model and will return the accuracy for each epoch of the CNN built model for the GTSRB dataset;

* train_yolov3.ipynb file, which is the custom notebook used to train the DNN YOLOv3 for our application;

And 2 scripts that are part of the demonstrator application:

* tsr4images.py file, used for detection and recognition on image files;

* tsr4videos.py file, used for detection and recognition on video streams;

There are 2 files saved for the trained models:

* yolov3_training_final.weights (combo with yolov3_testing.cfg and yolov3_training.cfg, the configuration files for YOLO), the model trained using YOLOv3 in 8000 iterations => accuracy 97.20%;

* TSR_model.h5, the model trained using our custom made CNN in 3 epochs with 2 iterations for each => accuracy 99.11%;

The classes for each model are described in theses files:

* classes_detection.names file for detection => 4 classes: prohibitory, danger, mandatory and other;

* classes_recognition.names file for recognition => 43 classes: from speed limit signs to priority and direction signs;

Demo output files of the app are saved into the demo folder.

## Download links

Download the GTSDB dataset: https://www.kaggle.com/safabouguezzi/german-traffic-sign-detection-benchmark-gtsdb

Download the GTSDB dataset in the YOLOv3 format: https://www.kaggle.com/valentynsichkar/traffic-signs-dataset-in-yolo-format

Download the GTSRB dataset: https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

Download the BelgiumTSC dataset: https://btsd.ethz.ch/shareddata/

Repository: https://github.com/Narcissimillus/TSR