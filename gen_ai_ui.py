import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

def select_video():
    # This function will allow the user to choose a video file and store the file path.
    filepath = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if filepath:
        video_var.set(filepath.split('/')[-1])  # Display only the filename
        print("Video selected:", filepath)
    else:
        print("No file selected")

def start_process():
    # This function prints the selected exercise and video file name when 'Start' is clicked.
    print("Selected exercise:", exercise_var.get())
    print("Video to process:", video_var.get())

# Create the main window
root = tk.Tk()
root.title("Exercise Video Uploader")
root.geometry("500x250")  # Width x Height

# Create a frame for some structure
frame = tk.Frame(root, padx=20, pady=20)
frame.pack(padx=10, pady=10)

# Dropdown menu for selecting the exercise type
exercise_var = tk.StringVar()
exercise_dropdown = ttk.Combobox(frame, textvariable=exercise_var, state="readonly")
exercise_dropdown['values'] = ('Deadlift', 'Bench Press', 'Squat', 'Other')
exercise_dropdown.current(0)
exercise_dropdown.grid(column=1, row=0, padx=10, pady=10)

# Label for dropdown
exercise_label = tk.Label(frame, text="Select Exercise:")
exercise_label.grid(column=0, row=0, sticky='W')

# Label for displaying selected video file
video_var = tk.StringVar()
video_label = tk.Label(frame, textvariable=video_var)
video_label.grid(column=1, row=2, padx=10, pady=10)

# Button to select video
select_button = tk.Button(frame, text="Select Video", command=select_video)
select_button.grid(column=0, row=2, padx=10, pady=10)

# Button to start the process
start_button = tk.Button(frame, text="Start", command=start_process)
start_button.grid(column=1, row=3, padx=10, pady=20)

# Start the GUI
root.mainloop()
