#Base Dataset

https://www.kaggle.com/jessicali9530/celeba-dataset

#Setup

Download the dataset and put all the jpg images in the data/faces folder

pip install -r requirements.txt

Install PyTorch:https://pytorch.org/get-started/locally/

#Run

Run main.py

#Result

Trad Machine: 96.47

Our Machine: 97.66

Our Machine With Relu: 97.81

#Transfer Learning

![Alt text](trans.png?raw=true "Title")

Facts:

1. Keep the penultimate layer

2. Freeze convolutional blocks

3. Depth augmentation (two new layers beyond pre-trained classification layer is cut-off point)
   
4. Layer-wise fine tuning (**not implemented**)

5. Normalisation/Regularisation - normalize and scale input activations of augmented
layers - L2/Batch (**not implemented**)

