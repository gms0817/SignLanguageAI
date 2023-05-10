import cv2
import mediapipe as mp
import time
import imutils


def detect_hands(img_frame):
    # Detect hands with mediapipe
    detected_hands = hands.process(cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB))

    return detected_hands


def detect_face(img_frame):
    # Detect face with media pipe
    detected_face = mpFace.process(cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB))

    return detected_face


def draw_hand_connections(img_frame, detected_hands):
    handType = 'TBD'
    if detected_hands.multi_hand_landmarks:
        for hand_landmarks in detected_hands.multi_hand_landmarks:
            # Check for both hands
            for hand in detected_hands.multi_handedness:
                if len(detected_hands.multi_handedness) == 2:
                    handType = 'Both'
                else:
                    # Check if left or right hand
                    handType = hand.classification[0].label

            for id, landmark in enumerate(hand_landmarks.landmark):
                h, w, c = img_frame.shape

                # Find coordinates of landmark(s)
                cx, cy = int(landmark.x * w), int(landmark.y * h)

                # Print landmark(s) for debugging
                landmark_id = mpHands.HandLandmark(id).name
                print(f'Hand(s): {handType} - Landmark ID: {landmark_id} - ({cx},{cy})')

                # Draw circle on landmark(s)
                cv2.circle(img_frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

                # Log to frame
                cv2.putText(img_frame, f'Hand(s): {handType}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 2, cv2.LINE_AA)

                # Draw landmark connection(s)
                mpDraw.draw_landmarks(img_frame, hand_landmarks, mpHands.HAND_CONNECTIONS)


def draw_face_boundary(img_frame, detected_face):
    if detected_face.detections:
        for face in detected_face.detections:
            mpDraw.draw_detection(img_frame, face)  # Draw boundary box around face


def main():
    # Initialize video feed
    video_feed = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        # Capture frame-by-frame
        ret, img_frame = video_feed.read()

        # Resize image
        img_frame = imutils.resize(img_frame, width=1000, height=1000)

        # Flip image
        img_frame = cv2.flip(img_frame, 1)

        # Detect hands and draw connections and box
        detected_hands = detect_hands(img_frame)
        draw_hand_connections(img_frame, detected_hands)

        # Detect face and draw box
        detected_face = detect_face(img_frame)
        draw_face_boundary(img_frame, detected_face)

        # Display output
        cv2.imshow('SL_AI - Realtime', img_frame)

        # Quit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    video_feed.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Configure mediapipe hands solution
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=False,  # Switch to true for image
        min_detection_confidence=0.3,
        max_num_hands=2)

    # Configure mediapipe face solution
    mpFace = mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5,
    )

    mpHolistic = mp.solutions.holistic
    mpDraw = mp.solutions.drawing_utils

    main()
