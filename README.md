# [PYTORCH] Build product classification model, automatically calculate product price
<p align="center">
 <h1 align="center">PriceVision</h1>
</p>

## Introduction
This is a project for automatically determining the price of each object.
* Determine the product name, price and quantity of each product, calculate the total product amount.
## Descriptions
* Collected custom dataset by recording individual product videos, then extracted frames for training.

* Trained a ResNet50-based classification model using PyTorch for identifying product types.

* Used YOLOv8 for object detection to locate products in video frames and passed bounding boxes through the classifier.

* Matched product names with corresponding prices stored in a JSON file to automatically display prices.

* Implemented calculation of total cost and product count.
</br>
<p align="center">
  <img src="demo/output.gif" width=600><br/>
  <i>Demo</i>
</p>


