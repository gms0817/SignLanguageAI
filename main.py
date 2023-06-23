import os
import threading
import cv2
import mediapipe as mp
import imutils
import itertools
import numpy as np
import csv
import tkinter as tk
import keyboard
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from models.ASL.ASL_Recognizer import ASLRecognizer
from models.BSL.BSL_Recognizer import BSLRecognizer


class MainApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        # Configure window
        self.title('SL_AI')

        # Create a container
        container = ttk.Frame(self, height=window_height, width=window_width)
        container.pack(side="top", fill="both", expand=True)

        # Configure frames from classes
        self.frames = {}

        def set_language(binding_event):
            slr.language_type = language_selection_dropdown.get()
            print(f'Debug: Language Type: {slr.language_type}')

        # Configure dropdown for language choice
        language_selection_dropdown = ttk.Combobox(
            state='readonly',
            values=['ASL', 'BSL'],
        )
        language_selection_dropdown.place(x=window_width - 200, y=20)
        language_selection_dropdown.current(0)

        # Bind language select
        language_selection_dropdown.bind('<<ComboboxSelected>>', set_language)
        for F in (HomePage, RealTimeRecognition, PhotoRecognition, VideoRecognition):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        # Show the home page
        self.show_frame(HomePage)

    def show_frame(self, controller):
        frame = self.frames[controller]
        frame.tkraise()
        frame.event_generate("<<ShowFrame>>")


class HomePage(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        # Define constants
        BODY_WIDTH = window_width - 300
        BUTTON_IPAD_X = 100
        LEFT_PADDING = 275

        # Title Text
        title_label = ttk.Label(self, text='Sign Language AI', font=("Arial", 25))
        title_label.pack(padx=LEFT_PADDING, pady=100)

        # Menu Buttons Housing
        menu_buttons_frame = ttk.Frame(self, width=BODY_WIDTH)
        menu_buttons_frame.pack(padx=LEFT_PADDING, pady=10, fill='both', expand=True)

        # Load images
        self.realtime_image = tk.PhotoImage(file="res/images/realtime.png").subsample(10, 10)
        self.photo_button = tk.PhotoImage(file="res/images/photo.png").subsample(10, 10)
        self.video_button = tk.PhotoImage(file="res/images/video.png").subsample(10, 10)

        # Real-Time Recognition
        realtime_button = ttk.Button(menu_buttons_frame, text="Realtime", image=self.realtime_image,
                                     command=lambda: controller.show_frame(RealTimeRecognition))
        realtime_button.pack(ipadx=BUTTON_IPAD_X, pady=10)

        realtime_label = ttk.Label(menu_buttons_frame, text='Real Time Recognition')
        realtime_label.pack()

        # Photo Recognition
        photo_button = ttk.Button(menu_buttons_frame, image=self.photo_button,
                                  command=lambda: controller.show_frame(PhotoRecognition))
        photo_button.pack(ipadx=BUTTON_IPAD_X, pady=10)

        photo_label = ttk.Label(menu_buttons_frame, text='Photo Recognition')
        photo_label.pack()

        # Video Recognition
        video_button = ttk.Button(menu_buttons_frame, image=self.video_button,
                                  command=lambda: controller.show_frame(VideoRecognition))
        video_button.pack(ipadx=BUTTON_IPAD_X, pady=10)

        video_label = ttk.Label(menu_buttons_frame, text='Video Recognition')
        video_label.pack()


class RealTimeRecognition(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller

        # Load in icon(s)
        self.home_icon = tk.PhotoImage(file='res/images/home.png').subsample(15, 15)

        # Open video stream
        self.video_feed = None

        # Label to house image frames from live feed
        self.feed_label = tk.Label(self)
        self.feed_label.pack(expand=True, fill='both')

        # Instruction label
        instruction_label = ttk.Label(self, text='Q: Quit - C: Capture Hand Landmarks')
        instruction_label.pack()

        # Home button
        home_button = ttk.Button(self, image=self.home_icon, width=10, command=self.go_home)
        home_button.place(x=30, y=15)

        # Bind video feed thread
        self.bind('<<ShowFrame>>', self.start_realtime_thread)

    def go_home(self):
        self.video_feed.release()
        print('Realtime Recognition Stopped.')
        self.controller.show_frame(HomePage)

    def start_realtime_thread(self, args):
        realtime_thread = threading.Thread(target=self.run_realtime_recognition)
        realtime_thread.isDaemon()

        try:
            realtime_thread.start()
            print('Realtime Recognition has started.')
        except RuntimeError:
            self.run_realtime_recognition()

    def run_realtime_recognition(self):
        # Start video feed
        self.video_feed = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if slr.language_type == 'ASL':
            class_num = len(slr.asl_recognizer_labels)
        elif slr.language_type == 'BSL':
            class_num = 1
            # class_num = len(slr.bsl_recognizer_labels)

        current_num_entries = 0
        MAX_ENTRIES = 50

        while self.video_feed.isOpened():
            # Capture frame-by-frame
            ret, live_layer = self.video_feed.read()

            # Resize image
            live_layer = imutils.resize(live_layer, width=window_width - 50, height=window_height - 50)

            # Flip image
            live_layer = cv2.flip(live_layer, 1)

            # Create no-background image
            detection_layer = np.empty(live_layer.shape)

            # Detect hands and draw connections and landmarks
            detected_hands = slr.detect_hands(live_layer)
            landmark_list = slr.draw_and_predict_hands(live_layer, detection_layer, detected_hands)

            # Detect face and draw box
            face_mesh = slr.detect_face(live_layer)
            slr.draw_face_mesh(live_layer, detection_layer, face_mesh)

            # Display output
            detection_layer = cv2.resize(detection_layer, dsize=(250, 250))  # Shrink for mini overlay
            live_layer[0: 250, 0: 250] = detection_layer

            # Process Key Input
            if keyboard.is_pressed('q'):
                self.video_feed.release()
                app.quit()
                break
            elif keyboard.is_pressed('c') and landmark_list is None:
                pass
            elif keyboard.is_pressed('c') and landmark_list is not None:
                if slr.language_type == 'ASL':
                    if current_num_entries == MAX_ENTRIES:
                        slr.export_hand_landmarks(class_num, current_num_entries, MAX_ENTRIES, slr.language_type,
                                                  landmark_list)
                        current_num_entries = 1
                        class_num += 1

                    else:
                        current_num_entries += 1
                        slr.export_hand_landmarks(class_num, current_num_entries, MAX_ENTRIES, slr.language_type,
                                                  landmark_list)
                elif slr.language_type == 'BSL' and detected_hands.multi_hand_landmarks:
                    # Extract landmarks for each hand
                    try:
                        multi_hand_landmarks = detected_hands.multi_hand_landmarks
                        left_hand_landmarks = slr.calculate_hand_landmarks_list(detection_layer, multi_hand_landmarks[0])
                        right_hand_landmarks = slr.calculate_hand_landmarks_list(detection_layer, multi_hand_landmarks[1])
                    except IndexError as e:
                        print('Number of Hands < 2!')

                    # Normalize each hand's landmarks
                    normalized_left_hand_landmarks = slr.normalize_landmark_list(left_hand_landmarks)
                    normalized_right_hand_landmarks = slr.normalize_landmark_list(right_hand_landmarks)
                    joined_landmarks = slr.join_multi_hand_landmarks(normalized_left_hand_landmarks,
                                                                     normalized_right_hand_landmarks)
                    if current_num_entries == MAX_ENTRIES:
                        # Export joint landmarks
                        slr.export_hand_landmarks(class_num, current_num_entries, MAX_ENTRIES, joined_landmarks)

                        current_num_entries = 1
                        class_num += 1
                    else:
                        current_num_entries += 1

                        # Export joint landmarks
                        slr.export_hand_landmarks(class_num, current_num_entries, MAX_ENTRIES, joined_landmarks)
                # Update visual info
                cv2.putText(live_layer,
                            f'Captured-Class: {class_num} Entry: {current_num_entries}/{MAX_ENTRIES}',
                            (50, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Update live layer feed
            display_image = cv2.cvtColor(live_layer, cv2.COLOR_BGR2RGB)  # Fix color prior to displaying
            display_image = Image.fromarray(display_image)
            display_image = ImageTk.PhotoImage(image=display_image)

            self.feed_label.configure(image=display_image)
            self.feed_label.image = display_image

        # When everything done, release the capture
        cv2.destroyAllWindows()


class PhotoRecognition(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        self.INITIAL_TEXT = "<<Select an image to upload>>"

        # Home button
        self.home_icon = tk.PhotoImage(file='res/images/home.png').subsample(15, 15)
        home_button = ttk.Button(self, image=self.home_icon, width=10, command=self.go_home)
        home_button.place(x=30, y=15)

        # Image frame
        self.file_path = None
        self.image = Image.open('res/images/placeholder.jpg')
        self.image = self.image.resize((400, 400), Image.Resampling.LANCZOS)
        self.image = ImageTk.PhotoImage(self.image)

        self.image_label = tk.Label(self, image=self.image)
        self.image_label.pack(padx=10, pady=(100, 20))

        # Prediction Label
        self.prediction_label = ttk.Label(self, text=self.INITIAL_TEXT, font=('Arial', 16))
        self.prediction_label.pack(side="top", padx=10, pady=10)

        # Select a photo
        self.browse_btn = ttk.Button(self, text="Browse", command=self.load_image)
        self.browse_btn.pack(ipadx=40, ipady=20, padx=10, pady=10)

        self.submit_btn = ttk.Button(self, text="Submit", command=self.classify_image)
        self.submit_btn.pack(ipadx=40, ipady=20, padx=10, pady=10)

    def go_home(self):
        # Reset image and labels
        self.image = Image.open('res/images/placeholder.jpg')
        self.image = self.image.resize((400, 400), Image.Resampling.LANCZOS)
        self.image = ImageTk.PhotoImage(self.image)
        self.image_label.config(image=self.image)

        self.prediction_label.config(text=self.INITIAL_TEXT)

        # go back to home screen
        self.controller.show_frame(HomePage)

    def load_image(self):
        # File chooser
        filetypes = (
            ('JPG', '*.jpg'),
            ('JPEG', '*.jpeg'),
            ('PNG', '*.png')
        )

        try:
            self.file_path = filedialog.askopenfilename(
                title="Select an image of Sign Language",
                initialdir='./',
                filetypes=filetypes
            )

            # Load the image file
            im = Image.open(self.file_path)

            im = im.resize((400, 400), Image.Resampling.LANCZOS)
            self.image = ImageTk.PhotoImage(im)

            # Update image display
            self.image_label.config(image=self.image)

            # Update prediction label text
            self.prediction_label.config(text=self.file_path[self.file_path.rindex('/') + 1:])
        except AttributeError:
            if self.file_path == '':
                self.file_path = None  # Ignore when file is not selected - e.g. cancel popup window

    def classify_image(self):
        if self.file_path is not None:
            # Make prediction
            temp_im = cv2.imread(self.file_path)
            prediction = slr.predict(temp_im)

            # Prep landmark image
            new_im = cv2.cvtColor(temp_im, cv2.COLOR_BGR2RGB)
            new_im = Image.fromarray(new_im)
            new_im = new_im.resize((400, 400), Image.Resampling.LANCZOS)

            # Update image and label
            self.image = ImageTk.PhotoImage(new_im)
            self.image_label.config(image=self.image)
            self.prediction_label.config(text=prediction)


class VideoRecognition(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.video_feed = None
        self.controller = controller

        # Label to house image frames from live feed
        self.place_holder_img = Image.open('res/images/placeholder.jpg')
        self.place_holder_img = self.place_holder_img.resize((window_width - 300, window_height - 300),
                                                             Image.Resampling.LANCZOS)
        self.place_holder_img = ImageTk.PhotoImage(self.place_holder_img)

        self.feed_label = tk.Label(self, image=self.place_holder_img)
        self.feed_label.pack(expand=True, fill='both')

        # Instruction label
        instruction_label = ttk.Label(self, text='Q: Quit')
        instruction_label.pack()

        # Home button
        self.home_icon = tk.PhotoImage(file='res/images/home.png').subsample(15, 15)
        home_button = ttk.Button(self, image=self.home_icon, width=10, command=self.go_home)
        home_button.place(x=30, y=15)

        # Select a video
        self.file_path = None
        self.browse_btn = ttk.Button(self, text="Browse", command=self.load_video)
        self.browse_btn.pack(ipadx=40, ipady=20, padx=10, pady=10)

        # Process the video
        self.process_video_btn = ttk.Button(self, text="Process Video", command=self.start_video_thread)
        self.process_video_btn.pack(ipadx=40, ipady=20, padx=10, pady=10)

    def go_home(self):
        # Reset page to default state
        self.file_path = None
        self.video_feed.release()
        self.feed_label.configure(image=self.place_holder_img)
        self.feed_label.image = self.place_holder_img

        # Go back to home page
        self.controller.show_frame(HomePage)

    def start_video_thread(self):
        video_thread = threading.Thread(target=self.process_video)
        video_thread.isDaemon()

        try:
            video_thread.start()
        except RuntimeError:
            self.process_video()

    def load_video(self):
        # File chooser
        filetypes = (
            ('MP4', '*.mp4'),
            ('MOV', '*.mov')
        )

        try:
            self.file_path = filedialog.askopenfilename(
                title="Select an image of Sign Language",
                initialdir='./',
                filetypes=filetypes
            )

        except AttributeError:
            if self.file_path == '':
                self.file_path = None  # Ignore when file is not selected - e.g. cancel popup window

    def process_video(self):
        # Load the video file
        self.video_feed = cv2.VideoCapture(self.file_path)

        prediction = ''
        while self.video_feed.isOpened():
            # Capture video frame-by-frame
            ret, video_frame = self.video_feed.read()

            # Resize video_frame
            if ret:
                video_frame = imutils.resize(video_frame, width=window_width - 50, height=window_height - 50)

                # Flip frame
                video_frame = cv2.flip(video_frame, 1)

                # Make prediction and draw landmarks if hand is detected
                previous_prediction = prediction
                prediction = slr.predict(video_frame)

                if prediction != previous_prediction and prediction != 'Unsure':
                    print(prediction, end=" ")

                if keyboard.is_pressed('q'):
                    app.quit()
                    break
                else:
                    cv2.waitKey(1)

                # Display image frame
                display_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)  # Fix color prior to displaying
                display_image = Image.fromarray(display_image)
                display_image = ImageTk.PhotoImage(image=display_image)

                # Update image display
                self.feed_label.configure(image=display_image)
                self.feed_label.image = display_image
            else:
                # Return to default image
                self.feed_label.configure(image=self.place_holder_img)
                self.feed_label.image = self.place_holder_img


class SignLanguageRecognition:

    def __init__(self):
        # Default Language type - global scope
        self.language_type = 'ASL'

        # Load classification models and labels
        self.MINIMUM_PREDICTION_CONFIDENCE = 0.10

        try:
            self.asl_recognizer = ASLRecognizer()
        except ValueError as e:
            print(f'Error: {e}')
        try:
            self.bsl_recognizer = BSLRecognizer()
        except ValueError as e:
            print(f'Error: {e}')
        self.asl_recognizer_labels = self.load_labels('ASL')
        self.bsl_recognizer_labels = self.load_labels('BSL')

        # Configure mediapipe hands solution
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,  # Switch to true for image
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            model_complexity=1
        )

        # Configure mediapipe face solution
        self.mpFaceMesh = mp.solutions.face_mesh
        self.mpFaceMeshImages = self.mpFaceMesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )

        self.mpHolistic = mp.solutions.holistic
        self.mpDraw = mp.solutions.drawing_utils
        self.mpDrawingStyles = mp.solutions.drawing_styles

    def predict(self, input_image):
        # Detect hands in image/frame
        detected_hands = self.detect_hands(input_image)

        # Detect face and draw box
        face_mesh = slr.detect_face(input_image)
        slr.draw_face_mesh(input_image, None, face_mesh)

        # Make prediction
        landmark_list = self.draw_and_predict_hands(input_image, None, detected_hands)

        # Make a prediction
        try:
            if self.language_type == 'ASL':
                prediction = self.asl_recognizer(landmark_list)
            elif self.language_type == 'BSL':
                prediction = self.bsl_recognizer(landmark_list)
            prediction_confidence = float(prediction[0])

            if prediction_confidence > self.MINIMUM_PREDICTION_CONFIDENCE:
                hand_sign_num = prediction[1]
                if self.language_type == 'ASL':
                    hand_sign_label = self.asl_recognizer_labels[hand_sign_num]
                elif self.language_type == 'BSL':
                    hand_sign_label = self.bsl_recognizer_labels[hand_sign_num]
            else:
                hand_sign_label = 'Unsure'
        except Exception as e:
            hand_sign_label = 'Unsure'
            pass
        return hand_sign_label

    def load_labels(self, language):
        label_list = []

        if language == 'ASL':
            filepath = 'models/ASL/ASL_classes.txt'
        elif language == 'BSL':
            filepath = 'models/BSL/BSL_classes.txt'
        else:
            return 'Invalid Language Type. Please review code.'

        try:
            with open(filepath, encoding='utf-8-sig') as file:
                label_list = csv.reader(file)
                label_list = [
                    row[0] for row in label_list
                ]
            print(f'{language} Labels: {label_list}')
        except FileNotFoundError:
            print('Unable to location ASL_classes. File may be missing or corrupt.')

        return label_list

    def calculate_bounding_box(self, live_layer, hand_landmarks):
        image_width, image_height = live_layer.shape[1], live_layer.shape[0]

        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(hand_landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv2.boundingRect(landmark_array)

        return [x, y, x + w, y + h]

    def draw_bounding_box(self, live_layer, landmarks, box_info_text):
        # Calculate bounding box per hand
        bounding_box = self.calculate_bounding_box(live_layer, landmarks)

        # Draw bounding box
        cv2.rectangle(live_layer, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (0, 0, 0), 1)
        cv2.putText(live_layer, box_info_text, (bounding_box[0] + 5, bounding_box[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    def normalize_landmark_list(self, landmark_list):  # compress landmarks to between -1 and 1 to normalize data
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

    def join_multi_hand_landmarks(self, left_hand_landmarks, right_hand_landmarks):
        joined_landmarks = []
        num_landmarks = len(left_hand_landmarks)

        # Join the left and right landmark lists per index
        for i in range(num_landmarks):
            difference = right_hand_landmarks[i] - left_hand_landmarks[i]
            # print(difference)
            joined_landmarks.append(difference)

        return joined_landmarks

    def detect_hands(self, live_layer):
        # Detect hands with mediapipe
        detected_hands = self.hands.process(cv2.cvtColor(live_layer, cv2.COLOR_BGR2RGB))

        return detected_hands

    def draw_and_predict_hands(self, input_image, detection_layer, detected_hands):
        landmark_list = None
        if detected_hands.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(detected_hands.multi_hand_landmarks, detected_hands.multi_handedness):
                # Construct landmark list for each hand
                landmark_list = self.calculate_hand_landmarks_list(input_image, hand_landmarks)
                # Debug Statement: print(landmark_list)

                # Normalize landmark list
                landmark_list = self.normalize_landmark_list(landmark_list)

                # Extract handtype
                handType = handedness.classification[0].label[0:]

                if len(detected_hands.multi_handedness) == 2:
                    handType = 'Both'  # 0 -> Left, 1 -> Right

                    # Extract landmarks for each hand
                    multi_hand_landmarks = detected_hands.multi_hand_landmarks
                    left_hand_landmarks = self.calculate_hand_landmarks_list(input_image, multi_hand_landmarks[0])
                    right_hand_landmarks = self.calculate_hand_landmarks_list(input_image, multi_hand_landmarks[1])

                    # Normalize each hand's landmarks
                    normalized_left_hand_landmarks = self.normalize_landmark_list(left_hand_landmarks)
                    normalized_right_hand_landmarks = self.normalize_landmark_list(right_hand_landmarks)
                    joined_landmarks = self.join_multi_hand_landmarks(normalized_left_hand_landmarks,
                                                                      normalized_right_hand_landmarks)

                    # Make prediction
                    prediction_confidence = 0.0
                    try:
                        if self.language_type == 'BSL':
                            prediction = self.bsl_recognizer(joined_landmarks)
                            prediction_confidence = float(prediction[0])

                            if prediction_confidence > self.MINIMUM_PREDICTION_CONFIDENCE:
                                hand_sign_num = prediction[1]
                                hand_sign_label = self.bsl_recognizer_labels[hand_sign_num]
                                print(f'BSL Prediction: {hand_sign_label}')
                        else:
                            hand_sign_label = 'Unsure'
                    except Exception as e:
                        hand_sign_label = 'Error. No label(s) found.'
                        print(f'Error: {e}')

                    # Draw landmark connection(s)
                    self.mpDraw.draw_landmarks(input_image, multi_hand_landmarks[0], self.mpHands.HAND_CONNECTIONS)
                    self.mpDraw.draw_landmarks(input_image, multi_hand_landmarks[1], self.mpHands.HAND_CONNECTIONS)

                    if detection_layer is not None:
                        self.mpDraw.draw_landmarks(detection_layer, multi_hand_landmarks[0], self.mpHands.HAND_CONNECTIONS)
                        self.mpDraw.draw_landmarks(detection_layer, multi_hand_landmarks[1], self.mpHands.HAND_CONNECTIONS)

                    # Draw bounding box
                    box_info_text = f'{self.language_type} | {handType} | Conf: {prediction_confidence}% | Sign:{hand_sign_label}'
                    self.draw_bounding_box(input_image, hand_landmarks, box_info_text)
                else:
                    for id, landmark in enumerate(hand_landmarks.landmark):
                        h, w, c = input_image.shape

                        # Find coordinates of landmark(s)
                        cx, cy = int(landmark.x * w), int(landmark.y * h)

                        # Draw circle on landmark(s)
                        cv2.circle(input_image, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

                        # Predict hand sign
                        prediction_confidence = 0.0

                        try:
                            if self.language_type == 'ASL':
                                prediction = self.asl_recognizer(landmark_list)
                                prediction_confidence = float(prediction[0])

                                if prediction_confidence > self.MINIMUM_PREDICTION_CONFIDENCE:
                                    hand_sign_num = prediction[1]
                                    hand_sign_label = self.asl_recognizer_labels[hand_sign_num]
                            else:
                                hand_sign_label = 'Unsure'
                        except Exception as e:
                            hand_sign_label = 'Error. No label(s) found.'
                            print(f'Error: {e}')

                        # Draw to frame
                        cv2.putText(input_image, f'Hand(s): {handType}', (750, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 0, 0), 2, cv2.LINE_AA)

                        # Draw landmark connection(s)
                        self.mpDraw.draw_landmarks(input_image, hand_landmarks, self.mpHands.HAND_CONNECTIONS)

                        if detection_layer is not None:
                            self.mpDraw.draw_landmarks(detection_layer, hand_landmarks, self.mpHands.HAND_CONNECTIONS)

                        # Draw bounding box
                        box_info_text = f'{self.language_type} | {handType} | Conf: {prediction_confidence}% | Sign:{hand_sign_label}'
                        self.draw_bounding_box(input_image, hand_landmarks, box_info_text)

            return landmark_list

    def calculate_hand_landmarks_list(self, live_layer, hand_landmarks):
        image_width, image_height = live_layer.shape[1], live_layer.shape[0]

        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(hand_landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    def export_hand_landmarks(self, class_num, current_num_entries, max_entries, landmark_list):
        print(f'Class Number: {class_num} Entries: {current_num_entries}/{max_entries}- Landmark List: {landmark_list}')

        directory_path = f'models/{self.language_type}'
        filepath = f'{directory_path}/{self.language_type}_landmarks.csv'
        directory = os.path.dirname(filepath)

        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(filepath, 'a', newline="") as file:
            print(*landmark_list)

            writer = csv.writer(file, delimiter=',')
            writer.writerow([class_num, *landmark_list])

    def detect_face(self, live_layer):
        # Detect face with media pipe
        detected_face = self.mpFaceMeshImages.process(live_layer)

        return detected_face

    def draw_face_mesh(self, live_layer, detection_layer, face_mesh):
        # Draw mesh
        if face_mesh.multi_face_landmarks:

            for face_landmarks in face_mesh.multi_face_landmarks:
                # Draw tessellations
                self.mpDraw.draw_landmarks(image=live_layer,
                                           landmark_list=face_landmarks,
                                           connections=self.mpFaceMesh.FACEMESH_TESSELATION,
                                           landmark_drawing_spec=None,
                                           connection_drawing_spec=self.mpDrawingStyles.get_default_face_mesh_tesselation_style())

                # Draw contours
                self.mpDraw.draw_landmarks(image=live_layer, landmark_list=face_landmarks,
                                           connections=self.mpFaceMesh.FACEMESH_CONTOURS,
                                           landmark_drawing_spec=None,
                                           connection_drawing_spec=self.mpDrawingStyles.get_default_face_mesh_contours_style())

                if detection_layer is not None:
                    # Draw tessellations

                    self.mpDraw.draw_landmarks(image=detection_layer,
                                               landmark_list=face_landmarks,
                                               connections=self.mpFaceMesh.FACEMESH_TESSELATION,
                                               landmark_drawing_spec=None,
                                               connection_drawing_spec=self.mpDrawingStyles.get_default_face_mesh_tesselation_style())
                    # Draw contours
                    self.mpDraw.draw_landmarks(image=detection_layer, landmark_list=face_landmarks,
                                               connections=self.mpFaceMesh.FACEMESH_CONTOURS,
                                               landmark_drawing_spec=None,
                                               connection_drawing_spec=self.mpDrawingStyles.get_default_face_mesh_contours_style())


if __name__ == '__main__':
    # Setup window dimensions
    window_width = 800
    window_height = 800

    # Instantiate SLR Class
    slr = SignLanguageRecognition()

    # Create app object
    app = MainApp()

    # Get screen dimensions
    screen_width = app.winfo_screenwidth()
    screen_height = app.winfo_screenheight()

    # Find the center point
    center_x = int(screen_width / 2 - window_width / 2)
    center_y = int(screen_height / 2 - window_height / 2)

    # Configure and run app
    app.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    app.resizable(width=False, height=False)  # Prevent Resizing
    app.mainloop()
