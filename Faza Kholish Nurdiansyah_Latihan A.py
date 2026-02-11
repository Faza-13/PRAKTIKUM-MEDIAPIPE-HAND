import cv2
import mediapipe

capture = cv2.VideoCapture(0)

mediapipehand = mediapipe.solutions.hands
tangan = mediapipehand.Hands(max_num_hands=2)
mpdraw = mediapipe.solutions.drawing_utils

while True:
    success, img = capture.read()
    if not success:
        print("loop jalan")
        break
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = tangan.process(imgRGB)
    if results.multi_hand_landmarks:
        print("tangan")
        for titiktangan in results.multi_hand_landmarks:
            mpdraw.draw_landmarks(img,titiktangan, mediapipehand.HAND_CONNECTIONS)
        if results.multi_handedness:
            for idx, hand in enumerate(results.multi_handedness):
                if hand.classification[0]:
                    if hand.classification[0].index == 1: #klasisfikasi tangan kanan
                        cv2.putText(img,"kanan",(200,50),cv2.FONT_HERSHEY_PLAIN,5,(255,0,0),3) #memberikan teks pada tampilan jendela dengan font harshey plain dengan fort size 5 dan ketebalan 5
                    elif hand.classification[0].index ==0 : #klasisfikasi tangan kiri
                        cv2.putText(img,"kiri",(200,50),cv2.FONT_HERSHEY_PLAIN,5,(0,0,255),3)

    else:
        print("tidak ada")
    cv2.imshow("webcam",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()


