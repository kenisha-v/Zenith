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
    feedback = process_video(video_var.get(), exercise_var.get())
    open_paragraph_window(feedback)

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

def process_frame_for_squats(landmarks, errors):
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
    
    if(abs(right_heel.y - right_toe.y) >= 0.05 or abs(left_heel.y - left_toe.y) >= 0.05):
        errors['feet_aligned'] = False
    
    if(not (optimal_torso_angle_range[0] <= right_torso_angle <= optimal_torso_angle_range[1]) or 
       not (optimal_torso_angle_range[0] <= left_torso_angle <= optimal_torso_angle_range[1])):
        errors['torso_aligned'] = False

    if(not (optimal_knee_angle_range[0] <= right_knee_angle <= optimal_knee_angle_range[1]) or 
       not (optimal_knee_angle_range[0] <= left_knee_angle <= optimal_knee_angle_range[1])):
        errors['knee_aligned'] = False

def process_frame_for_bench_press(landmarks, errors):
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
    
    if not left_wrist_in_line or not right_wrist_in_line:
        errors['wrist_aligned'] = False
    
    left_elbow_shoulder_distance = calculate_distance(left_elbow, left_shoulder)
    right_elbow_shoulder_distance = calculate_distance(right_elbow, right_shoulder)
    min_elbow_shoulder_distance = 0.035  # Adjust based on empirical observation
    
    if (left_elbow_shoulder_distance <= min_elbow_shoulder_distance or 
        right_elbow_shoulder_distance <= min_elbow_shoulder_distance):
        errors['elbow_aligned'] = False
    
    right_rep_range = calculate_angle(right_elbow, right_shoulder, right_hip)
    left_rep_range = calculate_angle(left_elbow, left_shoulder, left_hip)

    if  right_rep_range < 45 and left_rep_range < 45:
        right_chest_level = (right_shoulder.x + right_hip.x) / 2.5
        left_chest_level = (left_shoulder.x + left_hip.x) / 2.5
        
        if not (abs(right_wrist.x - right_chest_level) < 0.07 and abs(left_wrist.x - left_chest_level) < 0.07):
            errors['bar_aligned'] = False

def process_frame_for_deadlift(landmarks, initial_hip_position, errors):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    left_toe = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
    
    head = landmarks[mp_pose.PoseLandmark.NOSE.value]

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
    
    if(abs(right_knee.x > right_toe.x) > 0.01 or abs(left_knee.x > left_toe.x) > 0.01):
        errors['knee_toe_aligned'] = False
    
    if(left_spine_angle > 15 or right_spine_angle > 15):
        errors['spine_aligned'] = False

    if(abs(left_hip.y - initial_hip_position.y) > 0.1 or abs(right_hip.y - initial_hip_position.y) > 0.1):
        errors['hip_aligned'] = False

    if (optimal_hip_angle_range[0] > left_hip_angle or left_hip_angle > optimal_hip_angle_range[1] or 
        optimal_hip_angle_range[0] > right_hip_angle or right_hip_angle > optimal_hip_angle_range[1]):
        errors['hip_aligned'] = False

    if (optimal_knee_angle_range[0] > left_knee_angle or left_knee_angle > optimal_knee_angle_range[1] or 
        optimal_knee_angle_range[0] > right_knee_angle or right_knee_angle > optimal_knee_angle_range[1]):
        errors['knee_aligned'] = False

def get_ai_feedback_cohere(prompts):
    # print("AI Called: ", prompts)
    if prompts == "x":
         response = co.generate(model='command-xlarge-nightly', 
                                prompt="The user performed the exercise with optimal form. Provide words of encouragement",
                                max_tokens=300)
    else:
        response = co.generate(model='command-xlarge-nightly', 
                               prompt=f"The user is performing an exercise. Feedback: {prompts}. Give detailed advice on how to improve it. Dont give more than 3 points and be as direct and straight to the point as possible.",
                               max_tokens=300
        )
    # print(response.generations[0].text.strip)
    return response.generations[0].text.strip()

def exercise_choose(exercise, initial_hip_position, landmarks, errors):
    if exercise.lower() == "squats":
        process_frame_for_squats(landmarks, errors)

    if exercise.lower() == "bench press":
        process_frame_for_bench_press(landmarks, errors)

    if exercise.lower() == "deadlift":
        process_frame_for_deadlift(landmarks, initial_hip_position, errors)

def generate_feedback(exercise, errors):
    if exercise.lower() == "squats":
        prompts = "While performing squats, "
        if errors['feet_aligned']:
            prompts += "heels are coming off ground, "
        if errors['torso_aligned']:
            prompts += "the angle that my shoulder hips and knees make is not correct, "
        if errors['knee_aligned']:
            prompts += "the angle that my hips, knee and ankle make is not correct "
        if not errors['feet_aligned'] and not errors['torso_aligned'] and not errors['knee_aligned']:
            prompts = "x"

    if exercise.lower() == "bench press":
        prompts = "While performing a bench press, "
        if errors['wrist_aligned']:
            prompts += "wrist and elbow dont create a 90 degree angle with eachother, "
        if errors['elbow_aligned']:
            prompts += "the elbows flare out, "
        if errors['bar_aligned']:
            prompts += "the bar isn't positioned near the sternum "
        if not errors['wrist_aligned'] and not errors['elbow_aligned'] and not errors['bar_aligned']:
            prompts = "x"

    if exercise.lower() == "deadlift":
        prompts = "While performing a deadlift, "
        if errors['knee_toe_aligned']:
            prompts += "knees goes past the toes, "
        if errors['spine_aligned']:
            prompts += "spine is arched, "
        if errors['hip_aligned']:
            prompts += "the angle that my shoulder hips and knees make is not correct, "
        if errors['knee_aligned']:
            prompts += "the angle that my hips, knee and ankle make is not correct "
        if not errors['knee_toe_aligned'] and not errors['spine_aligned'] and not errors['hip_aligned'] and not errors['knee_aligned']:
            prompts = "x"
    
    feedback = get_ai_feedback_cohere(prompts)
    return feedback

def process_video(input_video_path, exercise):
    cap = cv2.VideoCapture(input_video_path)
    
    initial_hip_position = None

    # Initialize errors dictionary
    errors = {
        'feet_aligned': True,
        'torso_aligned': True,
        'knee_aligned': True,
        'wrist_aligned': True,
        'elbow_aligned': True,
        'bar_aligned': True,
        'knee_toe_aligned': True,
        'spine_aligned': True,
        'hip_aligned': True
    }

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = "output_video.mp4"
    out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

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

            exercise_choose(exercise, initial_hip_position, landmarks, errors)
        
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release video objects and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # print("Generating feedback")
    feedback = generate_feedback(exercise, errors)
    return feedback


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
