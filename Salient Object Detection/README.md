**ICCV Submission Salient Object Detection**

**Prerequistes**

1.Cuda Toolkit -10.1
2.PyTorch - min. 1.6
3.Torch Vision - 0.7
4.Detectron2
5.Google Colab
6.Opencv

**Train Model**

1.Download the model from https://jacobsuniversity-my.sharepoint.com/:f:/g/personal/dbohra_jacobs-university_de/EqUiMLX2j-ZEghWJfg1qG3oBhJnmojYgL8MZBXJDwcq31A?e=FZVMEC and weights from https://jacobsuniversity-my.sharepoint.com/:f:/g/personal/dbohra_jacobs-university_de/EirVVE-MtmtKjGSe3uXVNIIBdj6qYogNJW5NB1wsXd1chA?e=gjxnSL
2.To train the images please use the images under SCAS_Dataset/train/* all folders. Please make the corresponding changes file path changes in Train_SC_Model.py
3.In command line run the code as python Train_SC_Model.py

**Test Model**
1.Download the model from https://jacobsuniversity-my.sharepoint.com/:f:/g/personal/dbohra_jacobs-university_de/EqUiMLX2j-ZEghWJfg1qG3oBhJnmojYgL8MZBXJDwcq31A?e=FZVMEC and weights from https://jacobsuniversity-my.sharepoint.com/:f:/g/personal/dbohra_jacobs-university_de/EirVVE-MtmtKjGSe3uXVNIIBdj6qYogNJW5NB1wsXd1chA?e=gjxnSL
2.To test the images please use the images under SCAS_Dataset/val/* all folders. Please make the corresponding changes file path changes in the Test_SC_Model.py
3.In command line run the code as python Test_SC_Model.py

**Output**
Output would be visible under saliency prediction folder
