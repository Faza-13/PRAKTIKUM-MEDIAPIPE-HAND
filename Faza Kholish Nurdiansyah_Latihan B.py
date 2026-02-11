import cv2
import mediapipe

capture = cv2.VideoCapture(0)

mediapipehand = mediapipe.solutions.hands
tangan = mediapipehand.Hands(max_num_hands=2)
mpdraw = mediapipe.solutions.drawing_utils

while True:
    success, img = capture.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = tangan.process(imgRGB)

    if results.multi_hand_landmarks and results.multi_handedness:

        for idx, titiktangan in enumerate(results.multi_hand_landmarks):

            mpdraw.draw_landmarks(img, titiktangan, mediapipehand.HAND_CONNECTIONS)

            h, w, c = img.shape

            wrist = titiktangan.landmark[0]
            index_mcp = titiktangan.landmark[5]
            pinky_mcp = titiktangan.landmark[17]

            x0, y0 = int(wrist.x * w), int(wrist.y * h)
            x5, y5 = int(index_mcp.x * w), int(index_mcp.y * h)
            x17, y17 = int(pinky_mcp.x * w), int(pinky_mcp.y * h)

            cross = (x5 - x0)*(y17 - y0) - (y5 - y0)*(x17 - x0)

            # Ambil label tangan
            label = results.multi_handedness[idx].classification[0].label

            if label == "Right":
                kondisi = cross > 0
            else:  # Left
                kondisi = cross < 0

            if kondisi:
                posisi = "TELAPAK (DEPAN)"
                warna = (0, 255, 0)
            else:
                posisi = "PUNGGUNG (BELAKANG)"
                warna = (0, 0, 255)

            cv2.putText(img, posisi,
                        (x0, y0 - 20),
                        cv2.FONT_HERSHEY_PLAIN,
                        2,
                        warna,
                        2)

    else:
        print("tidak ada")

    cv2.imshow("webcam", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
