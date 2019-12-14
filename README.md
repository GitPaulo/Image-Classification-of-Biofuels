# Image-Classification-of-Biofuels-CNN
An analysis on the usefulness of various typical classification methods when applied to predicting wether images of common household processed materials classify as biomass. Classifiers used:

- VGG16
- Resnet50
- Xception
- KNN
- Random Forest

### Verdict
As predicted, when using simple traditional model fitting techniques, convolutional neural networks performed the best by a good margin.

<p align="center">
  <img src="./.github/test_averages.png"/>
</p

### Testing Application
A simple GUI based application was coded that allows the loading of the trained models to pedict the class on camera taken pictures. The application code can be found in the "app" folder.

<p align="center">
  <img src="./.github/example_app.gif"/>
</p>

#### Dataset and Trained Models
Large files can be found at: https://drive.google.com/open?id=1JPRsoK4WOJitXLvhHDIsXDuCVernQHh0
