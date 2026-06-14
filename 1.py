import cv2
import mediapipe as mp
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities
import math
import numpy as np

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

device = AudioUtilities.GetSpeakers()

volume = device.EndpointVolume

minVol, maxVol, _ = volume.GetVolumeRange()

print(minVol, maxVol)



while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:

        for handLms in results.multi_hand_landmarks:

            lmList = []

            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            if len(lmList) != 0:
                x1, y1 = lmList[4][1], lmList[4][2]  # thumb tip
                x2, y2 = lmList[8][1], lmList[8][2]  # index tip

            length = math.hypot(x2 - x1, y2 - y1)
            vol = np.interp(length, [20, 200], [minVol, maxVol])

            volume.SetMasterVolumeLevel(vol, None)
            

            mpDraw.draw_landmarks(
                img,
                handLms,
                mpHands.HAND_CONNECTIONS
            )

    cv2.imshow("Image", img)
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

