import cv2
import cohere
import tkinter as tk
import mediapipe as mp
import numpy as np
from tkinter import filedialog
from tkinter import ttk
from collections import namedtuple

co = cohere.Client('wwAmN0AxwrjsUdV7wUqBQUUVyCIi0n9TXLmAKSxL')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize the drawing utility
mp_drawing = mp.solutions.drawing_utils

Point = namedtuple('Point', ['x', 'y'])

def open_paragraph_window(feedback):
    new_window = tk.Toplevel(root)
    new_window.title("Processing Result")
    new_window.geometry("300x100")
    # Make the window not resizable
    new_window.resizable(False, False)
    # Aesthetic enhancements using a label
    result_label = tk.Label(new_window, text=feedback,
                            font=("Helvetica", 12), padx=10, pady=10)
    result_label.pack(expand=True)

def select_video():
    filepath = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if filepath:
        video_var.set(filepath)
        print("Video selected:", filepath)
    else:
        print("No file selected")

def start_process():
    prompt(video_var.get(), exercise_var.get())

def calculate_distance(a, b):
    return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle

def process_frame_for_squats(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    left_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value]
    left_toe = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
    
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    right_heel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value]
    right_toe = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]

    left_torso_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_torso_angle = calculate_angle(right_shoulder, right_hip, right_knee)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    
    optimal_torso_angle_range = (50, 180)
    optimal_knee_angle_range = (60, 180)
    
    if(abs(right_heel.y - right_toe.y) < 0.05 and abs(left_heel.y - left_toe.y) < 0.05):
        feet_aligned = True
    else:  
        feet_aligned = False
    
    if(optimal_torso_angle_range[0] <= right_torso_angle <= optimal_torso_angle_range[1] and
        optimal_torso_angle_range[0] <= left_torso_angle <= optimal_torso_angle_range[1]):
        torso_aligned = True
    else:    
        torso_aligned = False

    if(optimal_knee_angle_range[0] <= right_knee_angle <= optimal_knee_angle_range[1] and
        optimal_knee_angle_range[0] <= left_knee_angle <= optimal_knee_angle_range[1]):
        knee_aligned = True
    else:    
        knee_aligned = False

    return(feet_aligned, torso_aligned, knee_aligned)

def process_frame_for_bench_press(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    
    left_wrist_in_line = abs(left_wrist.x - left_elbow.x) < 0.05
    right_wrist_in_line = abs(right_wrist.x - right_elbow.x) < 0.05
    
    if left_wrist_in_line and right_wrist_in_line:
        wrist_aligned = True
    else:
        wrist_aligned = False
    
    left_elbow_shoulder_distance = calculate_distance(left_elbow, left_shoulder)
    right_elbow_shoulder_distance = calculate_distance(right_elbow, right_shoulder)
    min_elbow_shoulder_distance = 0.035  # Adjust based on empirical observation
    
    if (left_elbow_shoulder_distance > min_elbow_shoulder_distance and
        right_elbow_shoulder_distance > min_elbow_shoulder_distance):
        elbow_aligned = True
    else:
        elbow_aligned = False
    
    right_rep_range = calculate_angle(right_elbow, right_shoulder, right_hip)
    left_rep_range = calculate_angle(left_elbow, left_shoulder, left_hip)

    if  right_rep_range < 45 and left_rep_range < 45:
        right_chest_level = (right_shoulder.x + right_hip.x) / 2.5
        left_chest_level = (left_shoulder.x + left_hip.x) / 2.5
        
        if (abs(right_wrist.x - right_chest_level) < 0.07 and abs(left_wrist.x - left_chest_level) < 0.07):
            bar_aligned = True
        else:
            bar_aligned = False
        
    return(wrist_aligned, elbow_aligned, bar_aligned)

def process_frame_for_deadlift(landmarks, initial_hip_position):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    left_toe = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
    
    head = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]

    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    right_toe = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]
    
    left_spine_angle = calculate_angle(head, left_shoulder, left_hip)
    right_spine_angle = calculate_angle(head, right_shoulder, right_hip)

    left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    
    
    optimal_hip_angle_range = (45, 180)
    optimal_knee_angle_range = (80, 180)
    
    if(abs(right_knee.x > right_toe.x) > 0.01 and abs(left_knee.x > left_toe.x) > 0.01):
        knee_toe_aligned = False
    else:
        knee_toe_aligned = True
    
    if(left_spine_angle > 15 and right_spine_angle > 15):
        spine_aligned = False
    else:
        spine_aligned = True

    if(abs(left_hip.y - initial_hip_position.y) > 0.1 and abs(right_hip.y - initial_hip_position.y) > 0.1):
        hip_aligned = False
    else:
        hip_aligned = True

    if (optimal_hip_angle_range[0] <= left_hip_angle <= optimal_hip_angle_range[1] and
        optimal_hip_angle_range[0] <= right_hip_angle <= optimal_hip_angle_range[1]):
        hip_aligned = True
    else:   
        hip_aligned = False

    if(optimal_knee_angle_range[0] <= left_knee_angle <= optimal_knee_angle_range[1] and
        optimal_knee_angle_range[0] <= right_knee_angle <= optimal_knee_angle_range[1]):
        knee_aligned = True
    else:
        knee_aligned = False

    return(knee_toe_aligned, spine_aligned, hip_aligned, knee_aligned)


def get_ai_feedback_cohere(prompts):
    if prompts == " ":
         response = co.generate(model='command-xlarge-nightly', 
                                prompt="The user performed the exercise with optimal form. Provide words of encouragement",
                                max_tokens=300)
    else:
        response = co.generate(model='command-xlarge-nightly', 
                               prompt=f"The user is performing an exercise. Feedback: {prompts}. Give detailed advice on how to improve it. Dont give more than 3 points and be as direct and straight to the point as possible.",
                               max_tokens=300
        )
    return response.generations[0].text.strip()

def exercise_choose(exercise, initial_hip_position, landmarks):

    if exercise.lower() == "squats":
        prompts = "While performing squats, "
        output = process_frame_for_squats(landmarks) #func to squats
        if output[0]!=True:
            prompts += "heels are coming off ground, "
        if output[1] != True:
            prompts += "the angle that my shoulder hips and knees make is not correct, "
        if output[2] != True:
            prompts += "the angle that my hips, knee and ankle make is not correct "
        if output[0] and output[1] and output[2]:
            prompts = " "

    if exercise.lower() == "bench press":
        prompts = "While performing a bench press, "
        output = process_frame_for_bench_press(landmarks) #func to bench press
        if output[0]!=True:
            prompts += "wrist and elbow dont create a 90 degree angle with eachother, "
        if output[1] != True:
            prompts += "the elbows flare out, "
        if output[2] != True:
            prompts += "the  bar isn't positioned near the sternum "
        if output[0] and output[1] and output[2]:
            prompts = " "

    if exercise.lower() == "deadlift":
        prompts = "While performing a deadlift, "
        output = process_frame_for_deadlift(landmarks, initial_hip_position) #func to bench press
        if output[0]!=True:
            prompts += "knees goes past the toes, "
        if output[1] != True:
            prompts += "spine is arched, "
        if output[2] != True:
            prompts += "the angle that my shoulder hips and knees make is not correct, "
        if output[3] != True:
            prompts += "the angle that my hips, knee and ankle make is not correct "
        if output[0] and output[1] and output[2] and output[3]:
            prompts = " "
    
    feedback = get_ai_feedback_cohere(prompts)
    return feedback

def prompt(input_video_path, exercise):
    cap = cv2.VideoCapture(input_video_path)
    
    initial_hip_position = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            landmarks = results.pose_landmarks.landmark

            if initial_hip_position is None:
                initial_hip_position = Point(
                    (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x) / 2,
                    (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2
                )
            
            feedback = exercise_choose(exercise, initial_hip_position, landmarks)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release video objects and close windows
    cap.release()
    cv2.destroyAllWindows()
    open_paragraph_window(feedback)


if __name__ == "__main__":

    root = tk.Tk()
    root.title("Exercise Video Uploader")
    root.geometry("500x250")  # Width x Height

    frame = tk.Frame(root, padx=20, pady=20)
    frame.pack(padx=10, pady=10)

    exercise_var = tk.StringVar()
    exercise_dropdown = ttk.Combobox(frame, textvariable=exercise_var, state="readonly")
    exercise_dropdown['values'] = ('Deadlift', 'Bench Press', 'Squat', 'Other')
    exercise_dropdown.current(0)
    exercise_dropdown.grid(column=1, row=0, padx=10, pady=10)

    exercise_label = tk.Label(frame, text="Select Exercise:")
    exercise_label.grid(column=0, row=0, sticky='W')

    video_var = tk.StringVar()
    video_label = tk.Label(frame, textvariable=video_var)
    video_label.grid(column=1, row=2, padx=10, pady=10)

    select_button = tk.Button(frame, text="Select Video", command=select_video)
    select_button.grid(column=0, row=2, padx=10, pady=10)

    start_button = tk.Button(frame, text="Start", command=start_process)
    start_button.grid(column=1, row=3, padx=10, pady=20)

    root.mainloop()
