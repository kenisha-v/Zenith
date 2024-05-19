# Zenith

## Inspiration
Zenith was inspired by our own gym adventures. We saw beginners struggling with form, looking more like they were dancing than working out. We wanted to help them avoid turning their fitness journey into a blooper reel. So, we created Zenith to democratize access to expert fitness coaching and help everyone reach peak performance.

## What It Does
Zenith is your virtual gym buddy that never gets tired of correcting your form. Using MediaPipe for advanced motion tracking, Zenith watches your every move—like a fitness stalker, but in a good way. Then, it uses Cohere's generative AI to give you instant, personalized feedback on your squats, deadlifts, or bench presses.

## How We Built It
We mixed MediaPipe's tracking wizardry with Cohere's chatty AI, threw in some OpenCV for video processing, and wrapped it all in a Tkinter interface. This setup lets us spy on—I mean, analyze—your movements and provide feedback faster than you can say "gym selfie."

## Challenges We Ran Into
Finding the right generative AI was like searching for a needle in a haystack while blindfolded. We needed an AI that could not only talk the talk but also walk the walk—giving clear, actionable advice. After countless hours and more coffee than we'd like to admit, we found our match in Cohere's AI.

## Accomplishments That We're Proud Of
Picture this: 30 minutes before the deadline, everything goes haywire. Instead of panicking (okay, maybe a little), we channeled our inner superheroes and squashed the last-minute bug. We emerged victorious, with Zenith ready for launch and our sanity mostly intact.

## What We Learned
We learned that picking the right AI is like choosing the perfect avocado—tricky but totally worth it. After testing several AI models, Cohere’s AI won our hearts with its spot-on, clear feedback, essential for user trust and satisfaction.

## What's Next for Zenith
- **Real-Time Tracking:** We’re on a mission to make our tracking so good it feels like having a personal trainer ghosting your workout, giving immediate feedback to keep you on track.
- **Main Person Focus:** We’re developing advanced tracking algorithms to ensure Zenith keeps its eyes on you, even in a crowded gym.

## How to Run Zenith
To get started with Zenith, you'll need to install a few dependencies using `pip`, clone our repository, and then run the application. Here are the steps:

1. **Install Dependencies**:
   ```sh
   pip install tkinter opencv-python cohere mediapipe numpy

2. **Clone the repository**
   ```sh
    git clone <https://github.com/kenisha-v/Zenith.git>

3. **Run the Application:**
  ```sh
  cd <Zenith>
  python3 FitnessApp.py




