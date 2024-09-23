# Classifying images based on Age/Gender

This project is the individual project for 596E. It uses the [MiVolo](https://github.com/WildChlamydia/MiVOLO) model for bounding box predictions on images and then to classify whether the person(s) is a child or an adult and classifies their gender too.

**Requirements**

1. Download the files in this repo into a folder.
2. Download the `mivolo` folder on [this](https://github.com/WildChlamydia/MiVOLO) repo. Use the following link to quick download just the folder - [link](https://download-directory.github.io/?url=https%3A%2F%2Fgithub.com%2FWildChlamydia%2FMiVOLO%2Ftree%2Fmain%2Fmivolo)
3. Create a new subfolder called `models`. 
4. [Download](https://drive.google.com/file/d/1CGNCkZQNj5WkP3rLpENWAOgrBQkUWRdw/view) body + face detector model to `models/yolov8x_person_face.pt`
5. [Download](https://drive.google.com/file/d/11i8pKctxz3wVkDBlWKvhYIh7kpVFXSZ4/view) mivolo checkpoint to `models/mivolo_imbd.pth.tar`
6. Open a terminal window, navigate to this folder and run `pip install -r "requirements.txt"` to download the libraries required. 

**Basic Usage**

1. In the command line, run `python3 server.py` to run the server. 
2. Then run `python3 client.py --input <test_data>` where test_data is the folder containing input images you would like to classify.
3. `--output <outfile>` optional argument may be used to specify the output file name to store the response array as a npy file.

**Optional CLI arguments**

For client 

- `--single-person` optional argument may be used if it's known that each image contains only one person. It can help performance by ensuring single prediction per image.

for server

- `--device <device>` may be used to specify a different device to run the model on. Default is `cuda:0`.
- `--checkpoint <checkpoint>` may be used to specify a different mivolo model checkpoint (.pth.tar file) to use for classification.
- `--detector_weights <checkpoint>` may be used to specify a different yolo bounding box prediction model (.pt file) to use for classification.
