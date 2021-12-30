**ICCV3 Submission**
**EGNET**

**Prerequistes**
1. Cuda Toolkit -10.1
2. PyTorch - min. 1.6
3. Torch Vision - 0.7
4. Detectron2
5. Google Colab

**Train Model**
1. Download the weight and model from https://jacobsuniversity-my.sharepoint.com/:f:/g/personal/dbohra_jacobs-university_de/EhpQPjQ35VJOkFHpJ7PjNKABoR-Ru5ggILvZ1kxT-4xX8w?e=fl8s8B .
2. To train the images please use the images under DUTS-TR folder. Please make the corresponding changes file path changes in the run.py, dataset.py. 
3. In command line run the code as python run.py --mode train

**Test Model**
1. Download the weight and model from https://jacobsuniversity-my.sharepoint.com/:f:/g/personal/dbohra_jacobs-university_de/EhpQPjQ35VJOkFHpJ7PjNKABoR-Ru5ggILvZ1kxT-4xX8w?e=fl8s8B .
2. To test the images please use the images under DUTS-TR/DUTS-TR-Image/ folder. Please make the corresponding changes file path changes in the run.py, dataset.py. 
3. In command line run the code as python run.py --mode test --sal_mode p
