#Base Dataset

https://www.kaggle.com/jessicali9530/celeba-dataset

#Setup

Download the dataset and put all the jpg images in the data/faces folder

pip install -r requirements.txt

Install PyTorch:https://pytorch.org/get-started/locally/

#Run

Run main.py


#Transfer Learning

![Alt text](trans.png?raw=true "Title")

Facts:

1. Keep the penultimate layer

2. Freeze convolutional blocks

3. Depth augmentation (two new layers beyond pre-trained classification layer is cut-off point)
   
4. Layer-wise fine tuning (**not implemented**)

5. Normalisation/Regularisation - normalize and scale input activations of augmented layers 

##Referred Papers

1. Oquab, M., Bottou, L., Laptev, I., Sivic, J.: Learning and transferring mid-level
image representations using convolutional neural networks. Proceedings of the IEEE
conference on computer vision and pattern recognition, pp. 1717{1724 (2014). (**Discarding penultimate**)


2. Wang, Y., Ramanan, D., Hebert, M.: Growing a brain: Fine-tuning by increasing
model capacity. Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition, pp. 2471{2480 (2017). (**Discarding penultimate, L2-norm**)


3. Ioe, S., Szegedy, C.: Batch Normalization: Accelerating Deep Network Training
by Reducing Internal Covariate Shift. Proceedings of the 32Nd International Conference
on International Conference on Machine Learning, vol-37 (**Batch-norm**)

