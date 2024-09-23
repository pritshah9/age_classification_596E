import argparse
import os
import cv2
import torch
from mivolo.predictor import Predictor

from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.constants import DataTypes
from flask_ml.flask_ml_server.models import ResponseModel, TextResult, ImageResult


def get_images(folder_dir):
    images = []
    for image in os.listdir(folder_dir):
        if image.endswith(".jpg") or image.endswith(".jpeg") or image.endswith(".png"):
            images.append(os.path.join(folder_dir, image))
    return images


def get_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input", type=str, default=None, required=True, help="image file or folder with images")
    # parser.add_argument("--output", type=str, default=None, required=True, help="Folder for output results")
    parser.add_argument(
        "--detector-weights", default="models/yolov8x_person_face.pt", type=str
    )
    parser.add_argument("--checkpoint", default="models/mivolo_imbd.pth.tar", type=str)
    parser.add_argument("--with-persons", action="store_false")
    parser.add_argument("--disable-faces", action="store_true")
    parser.add_argument("--draw", action="store_false")
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--single-person", action="store_true")
    return parser


def get_bbdict_from_arr(arr):
    return {"X1": int(arr[0]), "Y1": int(arr[1]), "X2": int(arr[2]), "Y2": int(arr[3])}


def classify_given_age(age):
    return "child" if age <= 19 else "adult"


def update_params(params, new_params):
    for key, value in new_params.items():
        setattr(params, key, value)


parser = get_parser()
params = parser.parse_args()
predictor = Predictor(params, verbose=False)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

server = MLServer(__name__)


@server.route("/classify_age_gender", DataTypes.TEXT)
def process_text(inputs: list, parameters: dict) -> dict:
    main_res = []
    input_folder_dir = inputs[0].text
    images = get_images(input_folder_dir)
    update_params(params, parameters)
    single_person_flag = params.single_person
    l = len(images)
    c = 0
    no_predict = 0
    for image_name in images:
        avg_age = 0
        res = []
        img = cv2.imread(image_name)
        detected_objects, out_im = predictor.recognize(img)
        bboxes = detected_objects.yolo_results.boxes.xyxy.cpu().numpy()
        ages = detected_objects.ages
        genders = detected_objects.genders
        face_indexes = detected_objects.face_to_person_map.keys()
        for i in face_indexes:
            if ages[i] is not None:
                avg_age += ages[i]
                res.append(
                    {
                        "bbox": get_bbdict_from_arr(bboxes[i]),
                        "label": classify_given_age(int(ages[i])),
                        "gender": genders[i],
                    }
                )
        if len(res) == 0:
            no_predict += 1
            main_res.append(
                ImageResult(file_path=image_name, result=[{"no_predict": True}])
            )
            continue
        if single_person_flag:
            res = [res[0]]
            res[0]["label"] = classify_given_age(int(avg_age / len(face_indexes)))
        main_res.append(ImageResult(file_path=image_name, result=res))
    print("No predict: ", no_predict)
    response = ResponseModel(results=main_res, type=DataTypes.IMAGE)
    return response.get_response()


server.run(port=5000)
