# Brain Tumor CNN

To use this repository you will need to use Jupyter notebook and install the required libraries. Mainly pytorch and torchvision.

-

Included in this repository are the following files:

- tumor-detect.ipynb - Initial Model referred to as Model 1 in the paper with a simple CNN. This will load and train a model.
- tumor-detect-v2.ipynb - Model 2 in the paper based off VGG19. This will load and train a model.
- resnet-finetune.ipynb - This was Model 3 in the paper and was our failed attempt at transfer learning.
- make_validation_set.ipynb - The initial dataset didn't have a validation so this split the training set into a validation set.


You must run all the cells from top to bottom successfully to train then run a model.


For the purposes of testing this code, it's also hosted in a web wrapper online where you can test model 1 by uploading am image from the test set.


https://charlotte-ml-bt-6c169f565e86.herokuapp.com/