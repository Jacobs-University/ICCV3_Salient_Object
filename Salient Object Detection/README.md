ICCV Submission Salient Object Detection

Prerequistes

Cuda Toolkit -10.1
PyTorch - min. 1.6
Torch Vision - 0.7
Detectron2
Google Colab
Train Model

Download the weight and model from https://jacobsuniversity-my.sharepoint.com/:f:/g/personal/dbohra_jacobs-university_de/EhpQPjQ35VJOkFHpJ7PjNKABoR-Ru5ggILvZ1kxT-4xX8w?e=fl8s8B .
To train the images please use the images under DUTS-TR folder. Please make the corresponding changes file path changes in the run.py, dataset.py.
In command line run the code as python run.py --mode train
Test Model

Download the weight and model from https://jacobsuniversity-my.sharepoint.com/:f:/g/personal/dbohra_jacobs-university_de/EhpQPjQ35VJOkFHpJ7PjNKABoR-Ru5ggILvZ1kxT-4xX8w?e=fl8s8B .
To train the images please use the images under DUTS-TR/DUTS-TR-Image/ folder. Please make the corresponding changes file path changes in the run.py, dataset.py.
In command line run the code as python run.py --mode test --sal_mode p
