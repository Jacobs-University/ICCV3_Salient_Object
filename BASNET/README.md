**ICCV3 Submission**
**BASNET**

BASNet is a method for detecting and segmenting salient objects based on the simple yet successful UNet architecture, with a focus on increasing edge quality in segmentation masks.

**Prerequistes**
1. Detectron2
2. PyTorch - 1.6
3. Torch Vision - 0.7
5. Google Colab
6. Cuda ToolKit - 10.1

**Train the Model**
1. Run "Python basnet_train.py" in command line or google colab.Please make the changes to images directory and saved model in the code. The model basnet.pth https://jacobsuniversity-my.sharepoint.com/:u:/g/personal/dbohra_jacobs-university_de/EdC6k-1JoDFPqgBhhEqAN3MBk1qEauDwL88RaPValPXYZw?e=afaLc4 and save the path file as 'saved_models/basnet_bsi/basnet.pth'. 

**Test the Model**
1. Run the Python file with command as "Python basnet_test.py" or !python basnet_test.py in google colab. Please make the changes to images path and saved model in the code. The test images can be downloaded from the folder test_images.


