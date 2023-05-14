import cv2
import mediapipe as mp
import imutils
import itertools
import numpy as np


def detect_hands(live_layer):
    # Detect hands with mediapipe
    detected_hands = hands.process(cv2.cvtColor(live_layer, cv2.COLOR_BGR2RGB))

    return detected_hands


def calculate_hand_landmarks_list(live_layer, hand_landmarks):
    image_width, image_height = live_layer.shape[1], live_layer.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(hand_landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def normalize_landmark_list(landmark_list):  # compress landmarks to between -1 and 1 to normalize data
    # convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]  # base distance from wrist point

        landmark_list[index][0] = landmark_list[index][0] - base_x
        landmark_list[index][1] = landmark_list[index][1] - base_y
    # Convert to a one-dimensional list
    landmark_list = list(
        itertools.chain.from_iterable(landmark_list))

    # Normalization
    max_value = max(list(map(abs, landmark_list)))

    def normalize(value):
        return value / max_value

    landmark_list = list(map(normalize, landmark_list))

    return landmark_list


def draw_hand_connections(live_layer, detection_layer, detected_hands):
    handType = 'TBD'
    if detected_hands.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(detected_hands.multi_hand_landmarks, detected_hands.multi_handedness):
            # Construct landmark list for each hand
            landmark_list = calculate_hand_landmarks_list(live_layer, hand_landmarks)
            # Debug Statement: print(landmark_list)

            # Normalize landmark list
            normalized_landmark_list = normalize_landmark_list(landmark_list)
            # Debug Statement: print(normalized_landmark_list)

            # Extract handtype
            handType = handedness.classification[0].label[0:]

            if len(detected_hands.multi_handedness) == 2:
                handType = 'Both'

            for id, landmark in enumerate(hand_landmarks.landmark):
                h, w, c = live_layer.shape

                # Find coordinates of landmark(s)
                cx, cy = int(landmark.x * w), int(landmark.y * h)

                # Print landmark(s) for debugging
                landmark_id = mpHands.HandLandmark(id).name
                print(f'Hand(s): {handType} - Landmark ID: {landmark_id} - ({cx},{cy})')

                # Draw circle on landmark(s)
                cv2.circle(live_layer, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
                cv2.circle(detection_layer, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

                # Log to frame
                cv2.putText(live_layer, f'Hand(s): {handType}', (750, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 2, cv2.LINE_AA)

                # Draw landmark connection(s)
                mpDraw.draw_landmarks(live_layer, hand_landmarks, mpHands.HAND_CONNECTIONS)
                mpDraw.draw_landmarks(detection_layer, hand_landmarks, mpHands.HAND_CONNECTIONS)


def detect_face(live_layer):
    # Detect face with media pipe
    detected_face = mpFaceMeshImages.process(live_layer)

    return detected_face


def draw_face_mesh(live_layer, detection_layer, face_mesh):
    # Draw mesh
    if face_mesh.multi_face_landmarks:

        for face_landmarks in face_mesh.multi_face_landmarks:
            # Draw tessellations
            mpDraw.draw_landmarks(image=live_layer,
                                  landmark_list=face_landmarks, connections=mpFaceMesh.FACEMESH_TESSELATION,
                                  landmark_drawing_spec=None,
                                  connection_drawing_spec=mpDrawingStyles.get_default_face_mesh_tesselation_style())

            mpDraw.draw_landmarks(image=detection_layer,
                                  landmark_list=face_landmarks, connections=mpFaceMesh.FACEMESH_TESSELATION,
                                  landmark_drawing_spec=None,
                                  connection_drawing_spec=mpDrawingStyles.get_default_face_mesh_tesselation_style())

            # Draw contours
            mpDraw.draw_landmarks(image=live_layer, landmark_list=face_landmarks,
                                  connections=mpFaceMesh.FACEMESH_CONTOURS,
                                  landmark_drawing_spec=None,
                                  connection_drawing_spec=mpDrawingStyles.get_default_face_mesh_contours_style())

            mpDraw.draw_landmarks(image=detection_layer, landmark_list=face_landmarks,
                                  connections=mpFaceMesh.FACEMESH_CONTOURS,
                                  landmark_drawing_spec=None,
                                  connection_drawing_spec=mpDrawingStyles.get_default_face_mesh_contours_style())


def main():
    # Initialize video feed
    video_feed = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        # Setup size of window
        window_width = 1000
        window_height = 1000

        # Capture frame-by-frame
        ret, live_layer = video_feed.read()

        # Resize image
        live_layer = imutils.resize(live_layer, width=window_width, height=window_height)

        # Flip image
        live_layer = cv2.flip(live_layer, 1)

        # Create no-background image
        detection_layer = np.empty(live_layer.shape)

        # Detect hands and draw connections and box
        detected_hands = detect_hands(live_layer)
        draw_hand_connections(live_layer, detection_layer, detected_hands)

        # Detect face and draw box
        face_mesh = detect_face(live_layer)
        draw_face_mesh(live_layer, detection_layer, face_mesh)

        # Display output
        detection_layer = cv2.resize(detection_layer, dsize=(250, 250))  # Shrink for mini overlay
        live_layer[0: 250, 0: 250] = detection_layer

        cv2.imshow('SL_AI - Realtime', live_layer)

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
        max_num_hands=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    # Configure mediapipe face solution
    mpFaceMesh = mp.solutions.face_mesh
    mpFaceMeshImages = mpFaceMesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

    mpHolistic = mp.solutions.holistic
    mpDraw = mp.solutions.drawing_utils
    mpDrawingStyles = mp.solutions.drawing_styles

    main()
