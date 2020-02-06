Multi Label Classification
Daknet53 used as a base architecture and weights are pretrained on the imagenet Dataset which can be downloaded using official sites(links are provided below)
Classification based on regression.
Training is done on the 106 layer architecture .CFG file is provided for Architecture Configuration which can be build using Either
Pytorch or tensorflow.
For image and video  processing in detection we are using OPenCV.
Paffy can also be used to take input of videos from platforms such as youtube,Facebook..etc
here we are going to provide only detection code using pretrained weights and how to build the model using official CFG file.
Official weights are trained on the COCO dataset which have 80 common object classes.
You can run the above code in windows and ubuntu platforms.
You dont need any GPU for the above code to Labelling the input image or videos.It is working fine on i3 with 4 gb Ram.

Officials weight file link https://pjreddie.com/media/files/yolov3.weights
You have to download the weight file  and store in Configuration folder with a  name "network.weights".
Install all the libraries needed to run the code.
Make a Folder Names "output" for the output of images and videos
