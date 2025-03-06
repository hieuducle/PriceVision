# [PYTORCH] Build product classification model, automatically calculate product price
<p align="center">
 <h1 align="center">PriceVision</h1>
</p>

## Introduction
This is a project for automatically determining the price of each object.
* Determine the product name, price and quantity of each product, calculate the total product amount.
## Descriptions
* Building a product classification model with Resnet as backbone
* Use YOLO to detect the position of each item, then pass each item into a classification model to determine its label, and access a JSON file to retrieve the product price.
<p align="center">
  <img src="demo/output.gif" width=600><br/>
  <i>Camera app demo</i>
</p>


