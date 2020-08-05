# GlobalWheatObjectDetectionChallenge
Kaggle competition for wheat object detection in computer vision.
Project Summary
This project is used to detect wheat bounding boxes within each image as part of the "global-wheat-detection" Kaggle's competition.

At the beginning of this work, some installations are necessary to enable the running of the whole code, Then, a bunch of imports are needed from the same reason. In this project, we created a lot of deep learning models, including 10 Faster RCNN models (five networks use ResNet50 backbones and the other five use Vgg16 backbones), 5 Detectron2 models, and one YOLOv5 model. Every five models from each type of model described above was created by the same neural network architecture. The only difference between the five is that they were trained on a different set of images from the dataset, and evaluated on the complementary set of images, which is unique for each model, according to the 5-Fold cross-validation method.

In the preprocessing phase, we created a unique class called "WheatDataset" to support reading images from the global-wheat-detection dataset. This class is flexible and enables the "get_item" method to adjust the given image to each data structure used as an input to each model in this project. The WheatDataset has many options: reading the target labels either from a manipulated CSV file or from the original CSV file included in the dataset's directory, converting the images to the preferable type (RGB, BGR, GREY), dividing the images to the requested K-Folds of training and validation sets, if requested, etc. Next to this class, some functions used to create new datasets to get inputs for every model in this project. For example, the Faster RCNN models were trained on 512 X 512 sized images whether the Detectron2 and YOLOv5 models were trained on 1024 X 1024 sized images. We resized the images before the training phase to make the training process faster. Important to add that the YOLOv5 architecture was taken from the ultralytics GitHub repository, and excepts its input to include images and text files compatible with each target, and the data must be ordered uniquely in directories.

After getting the dataset ready for all of the models, we trained our Faster RCNNs using a step learning rate scheduler and a RadAm optimizer (20-22 epochs), our Detectron2 models using Focal Loss and step learning rate scheduler (15,000 iterations) and the YOLOv5 model (95 epochs). In the next phase of the project we created some functions to evaluate each model and to print F-Measure, True-Positive, False-Positive and False-Negative metrics, using some helper functions we created, as the "iou" function. Next, we wrote each F-Measure to a text file in order to use these values in the test phase as weights in an ensemble of models.

The last phase of this project is the test phase. In this phase, we created a function called "test_ensemble" to test all of the models against the test images. In this function we used TTA optimization which augmented each input image for several times, then we sent each augmented image through each model. After sending the different augmented images, the predicted targets were passed through another augmentation to convert them back to the origin. This method multiplied the number of predictions by the number of TTA augmentations (4 including the original targets). After passing each augmented/original image through each model, we mixed every prediction that belongs to the same family of models using WBF function (Weighted Box Fusion), an 'avg' conf_type, and the F_Measure weights. In more details, we mixed all of the 4 predictions derived from the TTA augmentations for each of the 10 different predictions of the faster RCNN models (Vgg16 / ResNet50 backbones), all of the 4 predictions derived from the TTA augmentations for each of the 5 different predictions of Detectron2 models, and all of the 4 predictions derived from the TTA augmentations using YOLOv5 model (1-Fold). In this period, we had three different predictions for the three "families of models" Faster RCNN, Detectron2, and YOLOv5 models. Then, We passed the three sub-ensemble of models through another run of the WBF function, but this time we used conf_type of 'max' prediction. Eventually, we got these last predictions of the ensemble, converted them to the format of Kaggle's results, and created the CSV submission file.

After reviewing the results on the validation set, the the YOLOv5 model was the most accurate model in terms of F-Measure value (0.95). The model's family with the lowest F-Measure value was the Detectron2 5-Fold models.

Detectron2 repo



YOLOv5 repo



Torchvision object detection fine tuning tutorial

