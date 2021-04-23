# comp9517-simPY-round-2

## Authors

* Evan Lee
* Theodore Koveos
* Ian Ng
* Rohan Maloney
* Andrew Timkov

## Instructions

### Distance and Velocity Estimation

* Copy `velocity.ipynb` into Google Colab (will not work on Jupyter Notebook)

* Setup a folder in your Google Drive with the path `Colab Notebooks/comp9517/`

* In this folder ensure there are the following files:

  * `condensed_data.zip` - a zip folder with all of the velocity training and testing data folders
  * `mask_rcnn_coco.h5` - pretrained weights which can be downloaded from [here](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5)
  * `car_mask_rcnn.h5` - pretrained weights fitted for the TuSimple dataset. This is optional as the model training will create this file
  * `mrcnn` folder - contains source code for [Matterport's Mask R-CNN implementation](https://github.com/matterport/Mask_RCNN/) with some edits
  * `evaluate` folder - contains source code for [TuSimple's evaluation code](https://github.com/TuSimple/tusimple-benchmark/tree/master/evaluate) with some edits

* Run all code cells

* Execution calls has the structure of `make_predictions(TESTING_DIR, detection_mode=["distance", "velocity"], show_img=True, show_all=False, my_evaluate=True, save=True, subset=[10, 23])`

  * `TESTING_DIR` - directory with testing data
  * `detection_mode` - `distance` to calculate distance, `velocity` to calculate velocity
  * `show_img` - set to True if you want the images to be shown
  * `show_all` - set to True if you want False positives (un-annotated vehicles) to be shown
  * `my_evaluate` - set to True if you want each input to be evaluated for accuracy
  * `save` - set to True if you want images to be saved to a file
  * `subset` - define a list of clip number that you want to run predictions for. If this is set to `[]` or ommitted, then predictions will be run for every clip in `TESTING_DIR`

### Lane Detection
* Open `lane_detection.ipynb `
* choose which dataset you wish to run the code on in cell 2
* choose which clip from the dataset you would like to run the code on