'''Script to classify faces in video'''
# Extends tutorial from [Neural Nine](https://www.youtube.com/watch?v=pQvkoaevVMk)

import argparse
import json
from pathlib import Path
import threading

import cv2


parser = argparse.ArgumentParser(description ='Detect face matches in video stream.')
parser.add_argument("-c", "--config", help="JSON with names and reference images", required=True)
parser.add_argument("-v", "--video", help="Video as data source")
args = parser.parse_args()


def valid_config(file_data: dict) -> bool:
    '''Validates data from config file'''
    # could instead return boolean of valid and allow program to continue w/o invalid
    for reference_image_path in file_data.values():
        temp_path = Path(reference_image_path)
        if not temp_path.exists():
            print(f"'{temp_path}' not valid")
            return False
    return True

def parse_config(file_data: dict) -> tuple[dict, dict]:
    '''Parses values from config file'''
    matches_dict = dict.fromkeys(file_data.keys(), False) # initialize names to not found
    images_dict = {name: cv2.imread(image_path) for name, image_path in file_data.items()}

    return matches_dict, images_dict


config_path = Path(args.config)
if not config_path.exists():
    print(f"{config_path} does not exist")
    raise SystemExit(1)

# Validates configuration file
with open(config_path, mode='r', encoding='utf-8') as config_fp:
    config = json.load(config_fp)
    # could move checking config path by excepting failed read
if not valid_config(config):
    print(f"{config_path} is not valid")
    raise SystemExit(1)

if args.video:
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"{video_path} does not exist")
        raise SystemExit(1)
    cap = cv2.VideoCapture(args.video)
else:
    cap = cv2.VideoCapture(0)

# Placed later to follow argument checks first
from deepface import DeepFace


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0

face_match, reference_images = parse_config(config)

assert face_match.keys() == reference_images.keys()

def check_face(local_frame, reference_face, name: str) -> None:
    '''Checks if face is detected in frame'''
    global face_match
    try:
        face_match[name] = DeepFace.verify(local_frame, reference_face.copy())['verified']
    except ValueError:
        face_match[name] = False

while True:
    ret, frame = cap.read()

    if ret:
        if counter % 90 == 0:
            try:
                for person_name, ref_image in reference_images.items():
                    threading.Thread(target=check_face, args=(frame.copy(), ref_image.copy(),
                                                          person_name),).start()
            except ValueError:
                pass
        counter += 1

        found_names = [name for name, found in face_match.items() if found]
        if found_names:
            cv2.putText(frame, f"Found: {', '.join(found_names)}", (20, 430),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "Nobody knows", (20, 430),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow('Camera Feed', frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cv2.destroyAllWindows()
