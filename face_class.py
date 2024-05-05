'''Script to classify faces in video'''
# Extends tutorial from [Neural Nine](https://www.youtube.com/watch?v=pQvkoaevVMk)

import argparse
import threading
import cv2
from deepface import DeepFace


# cap = cv2.VideoCapture(video)

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


counter = 0

people = ['p1', 'p2', 'p3']

face_match = dict.fromkeys(people, False)

references_images = {
    'p1': cv2.imread("1.jpg"),
    'p2': cv2.imread("2.jpeg"),
    'p3': cv2.imread('3.jpeg'),
}

assert face_match.keys() == references_images.keys()

def check_face(local_frame, reference_face, name: str):
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
                for person_name, ref_image in references_images.items():
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
