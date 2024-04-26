import cv2
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from math import sqrt

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

mp_drawing = mp.solutions.drawing_utils

# Initialisation de la vidéo
cap = cv2.VideoCapture(0)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Conversion de l'image en RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Afficher les landmarks de la main
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                       mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                                       mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
            

            # Calcul de la position du pouce et de l'index
            thumb_y = hand_landmarks.landmark[4].y

            # Positions des autres doigts
            index_y = hand_landmarks.landmark[8].y
            middle_y = hand_landmarks.landmark[12].y
            ring_y = hand_landmarks.landmark[16].y
            pinky_y = hand_landmarks.landmark[20].y



            wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
            wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y

            # Position des landmarks des doigts
            finger_x = [hand_landmarks.landmark[i].x for i in range(20)]
            finger_y = [hand_landmarks.landmark[i].y for i in range(20)]

            # Calculer la distance moyenne entre les landmarks des doigts et le poignet
            distance_to_wrist = sum([sqrt((x - wrist_x)**2 + (y - wrist_y)**2) for x, y in zip(finger_x, finger_y)]) / len(finger_x)

            
            if hand_landmarks.landmark[8].y > hand_landmarks.landmark[4].y and hand_landmarks.landmark[12].y < hand_landmarks.landmark[11].y or distance_to_wrist < 0.1:
                sign = "👊POING "  # Emoji pour poing fermé

            elif thumb_y > index_y and thumb_y > pinky_y and thumb_y > middle_y:
                sign = "MAIN OUVERTE"

            elif  thumb_y < index_y and thumb_y < middle_y and thumb_y < ring_y and thumb_y < pinky_y:
                sign = "👍"

            elif middle_y < thumb_y and middle_y < index_y and middle_y < ring_y and middle_y < pinky_y:
                sign = "INDECENT"

            else:
                sign = "none"

            # Création de l'image avec l'emoji
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            font = ImageFont.truetype("seguiemj.ttf", 50)  # Utilisation d'une police prenant en charge les emojis
            draw.text((50, 50), sign, font=font, fill=(255, 0, 0))  # Remarque: rouge est (B, G, R)

            # Convertir l'image PIL en image numpy pour l'affichage
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    cv2.imshow('Hand Sign Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
