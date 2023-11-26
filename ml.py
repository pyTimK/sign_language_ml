
#! 1. Import and Install Dependencies

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp


#! 2. Keypoints using MP Holistic

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connection




def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
                             
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 



#! 3. Extract Keypoint Values
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])




#! 4. Setup Folders for Collection
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Is begninner or intermediate
is_beginner = False

# Actions that we try to detect
actions = np.array([
    '0_none',
    '1_yellow',
    '2_continue',
    '3_green',
    '4_black',
    '5_gray',
    '6_library',
    '7_cr',
    '8_church',
    '9_hospital',
    '10_home',
    '11_tomorrow',
    '12_feelings',
    '13_sad',
    '14_happy',
    '15_sick',
    '16_sister',
    '17_relatives',
    '18_grandpa',
    '19_father',
    '20_today',
]) if is_beginner else np.array([
    '0_none',
    '1_my_favorite_color_is_green',
    '2_what_color_do_you_want',
    '3_whats_the_color_of_the_shoes',
    '4_whats_the_color_of_the_thsirt',
    '5_are_you_okay',
    '6_im_excited',
    '7_what_happened',
    '8_where_does_it_hurt',
    '9_why_are_you_sad',
    '10_are_your_parents_strict',
    '11_call_your_sister_now',
    '12_how_are_your_parents',
    '13_my_family_consists_of_six',
    '14_where_is_your_brother',
    '15_can_you_help_me',
    '16_nice_to_meet_you',
    '17_see_you_later',
    '18_what_time_is_it',
    '19_ill_make_breakfast',
    '20_where_are_you_going',
])




# Number of videos per action
no_sequences = 50

# Number of frames per video
sequence_length = 30 if is_beginner else 40

# Folder start
start_folder = 1


def setup_folder_for_collection():
  for action in actions: 
      # dirmax = np.max(np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int))
      for sequence in range(1,no_sequences+1):
          try: 
              os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
              # os.makedirs(os.path.join(DATA_PATH, action, str(dirmax+sequence)))
          except:
              pass




#! 5. Collect Keypoint Values for Training and Testing
import math
def create_training_data():
  cap = cv2.VideoCapture(1)
  # Set mediapipe model 
  with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
      
      # NEW LOOP
      # Loop through actions
      for action in actions:
          # Loop through sequences aka videos
          for sequence in range(start_folder, start_folder+no_sequences):
              # Loop through video length aka sequence length
              for frame_num in range(sequence_length):

                  # Read feed
                  ret, frame = cap.read()

                  # Make detections
                  image, results = mediapipe_detection(frame, holistic)

                  # Draw landmarks
                  draw_styled_landmarks(image, results)

                  # Draw frame percet
                  _percent = math.floor(frame_num * 100 / sequence_length)
                  cv2.rectangle(image, (0,0), (200, 10), (0, 0, 0), -1)
                  cv2.rectangle(image, (0,0), (math.floor((200/100) * _percent), 10), (0, 250, 0), -1)

                  # Draw videos percet
                  _percent = math.floor(sequence * 100 / no_sequences+start_folder)
                  cv2.rectangle(image, (400,0), (600, 10), (0, 0, 0), -1)
                  cv2.rectangle(image, (400,0), (400 + math.floor((200/100) * _percent), 10), (100, 100, 255), -1)
                  
                  # NEW Apply wait logic
                  if frame_num == 0: 
                      cv2.putText(image, f'GET READY: {action}-{sequence}', (120,200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                      cv2.putText(image, '{}  -  {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                      # Show to screen
                      cv2.imshow('OpenCV Feed', image)
                      if sequence == 1:
                        cv2.waitKey(3000)
                  else: 
                      cv2.putText(image, '{}  -  {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                      
                      

                      # cv2.putText(image, f'{(math.floor(frame_num * 100 / sequence_length))}%', (120,100), 
                      #           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                      

                      # Show to screen
                      cv2.imshow('OpenCV Feed', image)
                  
                  # NEW Export keypoints
                  keypoints = extract_keypoints(results)
                  npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                  np.save(npy_path, keypoints)

                  # Break gracefully
                  if cv2.waitKey(10) & 0xFF == ord('q'):
                      break
                      
      cap.release()
      cv2.destroyAllWindows()






#! 6. Preprocess Data and Create Labels and Features
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def preprocess():
  label_map = {label:num for num, label in enumerate(actions)}

  sequences, labels = [], []
  for action in actions:
      for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
          window = []
          for frame_num in range(sequence_length):
              res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
              window.append(res)
          sequences.append(window)
          labels.append(label_map[action])

  X = np.array(sequences)
  y = to_categorical(labels).astype(int)
  # print(f"sequences SHAPE: {np.array(sequences).shape}")
  # print(f"labels SHAPE: {np.array(labels).shape}")
  print(f"X SHAPE: {X.shape}")
  # print(f"y: {y}")
  # exit()
  return train_test_split(X, y, test_size=0.2)



#! 7. Build and Train LSTM Neural Network
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

def build_model(epochs: int):
  X_train, X_test, y_train, y_test = preprocess()

  log_dir = os.path.join('Logs')
  tb_callback = TensorBoard(log_dir=log_dir)


  #   model = Sequential()
  #   model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(40,1662)))
  #   model.add(LSTM(128, return_sequences=True, activation='relu'))
  #   model.add(LSTM(64, return_sequences=False, activation='relu'))
  #   model.add(Dense(64, activation='relu'))
  #   model.add(Dense(32, activation='relu'))
  #   model.add(Dense(actions.shape[0], activation='softmax'))

  
  model = Sequential()

  # Convolutional layers
  model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(sequence_length, 1662)))
  model.add(MaxPooling1D(pool_size=2))
  model.add(Conv1D(128, kernel_size=3, activation='relu'))
  model.add(MaxPooling1D(pool_size=2))

  # LSTM layers
  model.add(LSTM(64, return_sequences=True, activation='relu'))
  model.add(LSTM(64, return_sequences=False, activation='relu'))

  # Fully connected layers
  model.add(Dense(64, activation='relu'))
  model.add(Dropout(0.5))  # Add dropout layer
  model.add(Dense(32, activation='relu'))
  model.add(Dropout(0.5))  # Add dropout layer

  # Output layer
  model.add(Dense(len(actions), activation='softmax'))  # Assuming 2 classes: "None" and "Yellow"

  model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

  #? Define early stopping callback
  #   early_stopping = EarlyStopping(monitor='val_loss',  # Choose the metric to monitor
  #                             #    patience=10,          # Number of epochs with no improvement after which training will be stopped
  #                             #    min_delta=0.01,      # Minimum change in the monitored quantity to qualify as an improvement
  #                             #    baseline=0.90,        # Stop training when the accuracy reaches this baseline value
  #                                restore_best_weights=False)  # Restore model weights from the epoch with the best value of the monitored quantity

  checkpoint = ModelCheckpoint("model.keras", 
                    monitor="val_loss", mode="min", 
                    save_best_only=True, verbose=1)

  model.fit(X_train, y_train, epochs=epochs, callbacks=[tb_callback, checkpoint], validation_data=(X_test, y_test))

  #   model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])
  model.save('model_final.keras')
  



#! 8. Testing
# from scipy import stats
colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    _y = 17
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,75+num*_y), (int(prob*100), 90+num*_y), colors[0], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv2.LINE_AA)
        
    return output_frame


# plt.figure(figsize=(18,18))
# plt.imshow(prob_viz(res, actions, image, colors))
# 1. New detection variables
def start():
  sequence = []
  predictions = []
  threshold = 0.5
  model = load_model('model.keras')

  cap = cv2.VideoCapture(1)
  # Set mediapipe model 
  with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
      while cap.isOpened():

          # Read feed
          ret, frame = cap.read()

          # Make detections
          image, results = mediapipe_detection(frame, holistic)
          # print(results)
        #   print(predictions)
          
          # Draw landmarks
          draw_styled_landmarks(image, results)
          
          # 2. Prediction logic
          keypoints = extract_keypoints(results)
          sequence.append(keypoints)
          sequence = sequence[-1*sequence_length:]
          
          if len(sequence) == sequence_length:
              res = model.predict(np.expand_dims(sequence, axis=0))[0]
              print(actions[np.argmax(res)])
              predictions.append(np.argmax(res))
              

              # Viz probabilities
              image = prob_viz(res, actions, image, colors)
              
          cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)

          # Show to screen
          cv2.imshow('OpenCV Feed', image)

          # Break gracefully
          if cv2.waitKey(10) & 0xFF == ord('q'):
              break
      cap.release()
      cv2.destroyAllWindows()



#? WHAT TO DO
# setup_folder_for_collection()
# create_training_data()
build_model(350)
# start()

