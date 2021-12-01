# defHTR

Pytorch code for deformation-invariant line-level Handwritten Text Recognition, as proposed in [paper](https://www.cs.uoi.gr/~sfikas/21Retsinas-Deformation_invariant_networks[ICIP].pdf) (accepted to ICIP'21).

**Motivation:**
Image deformations under simple geometric restrictions are crucial for Handwriting Text Recognition (HTR), since different writing styles can be viewed as simple geometrical deformations of the same
textual elements. 
<br>
**Contibutions:** 1) Exploration of different existing strategies for ensuring deformation invariance, including spatial transformers and deformable convolutions, under the
context of text recognition. 
2) Introduction of a new deformation-based algorithm, inspired by adversarial learning, which aims to reduce character output uncertainty during evaluation time. 

**DNN Architecture:** A Convolutional-only HTR system is presented (see [paper](https://www.cs.uoi.gr/~sfikas/21Retsinas-Deformation_invariant_networks[ICIP].pdf)), 
where the output of a convolutional backbone, which transforms the images into a sequence of feature vectors, is fed into a cascade of 1-D convolutional layers.
Model architecture can be modified by changing the the cnn_cfg and cnn_top variables in config.py.
Specifically, CNN backbone is consisted of multiple stacks of ResBlocks and the default setting `cnn_cfg = [(2, 32), 'M', (4, 64), 'M', (6, 128), 'M', (2, 256)]` is interpeted as follows:
the first stack consists of 2 resblocks with output channels of 32 dimensions, the second of 4 resblocks with 64 output channels etc. 
The head, consisted of three 1-D convolutional layers, can be  modified through the cnn_top variable, which controls the number of output channels in these layers.
 
Selected Features:
* Dataset is saved in a '.pt' file after the initial preprocessing for faster loading operations
* All images are resized to 128x1024 (using padding if possible in order to retain aspect ratio).
* Transformations used during training: global affine, local deformations (elastic net), local morphological operations.

File valid_deforms.py contains: 
*  local_deform(img): creates a grid of smooth offsets in order to perform an elastic net distortion.
*  morphological(img): creates a grid which control the morhpological operation to apply in each neighborhod.
*  uncertainty_reduction(net, img):  the proposed uncertainty reduction method to be used at evaluation. It discovers small local deformations (akin to local_deform method) which maximize the per character softmax output, i.e. minimize the uncertainty to select these characters.  

**Installation:** 

```
python3 -m venv venv_defhtr         # Create a virtual environment.. (optional)
source venv_defhtr/bin/activate     # ..and activate the virtual environment (optional)
pip3 install --upgrade pip          # Upgrade pip
pip3 install -r requirements.txt    # Install required libraries
cat utils/iam_config.py             # Update the folder where IAM resides in your hard drive
python3 train_htr.py                # Have fun!!
```

**Note:** Local paths of IAM dataset (https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) are hardcoded in ``utils/iam_config.py``


