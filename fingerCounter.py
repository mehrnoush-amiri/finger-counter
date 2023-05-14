import cv2
import mediapipe as mp

cam = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
fingerCoordinates = [(8, 6), (12, 10), (16, 14), (20, 18)]
thumbCoordinate=(4,2)

while True:
    success, img=cam.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    multiLandMarks = results.multi_hand_landmarks

    if multiLandMarks:
        handpoints = []
        for handLms in multiLandMarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            for idx, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                handpoints.append((cx, cy))
            for point in handpoints:
                cv2.circle(img, point, 10, (255,0,255), cv2.FILLED)
            upCount = 0
            for coordinate in fingerCoordinates:
                if handpoints[coordinate[0]][1] < handpoints[coordinate[1]][1]:
                    upCount += 1
            if handpoints[thumbCoordinate[0]][0] > handpoints[thumbCoordinate[1]][0]:
                upCount +=1

            cv2.putText(img, str(upCount), (150,150), cv2.FONT_HERSHEY_PLAIN, 12, (255, 0, 12), 12)

    cv2.imshow("Finger counter", img)
    cv2.waitKey(2)
