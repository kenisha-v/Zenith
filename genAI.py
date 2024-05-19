import cohere

co = cohere.Client('wwAmN0AxwrjsUdV7wUqBQUUVyCIi0n9TXLmAKSxL')

# gives feedback to fix form using Cohere 
def get_ai_feedback_cohere(prompts):
    response = co.generate(
        model='command-xlarge-nightly',  # Use an available model name
        prompt=f"The user is performing an exercise. Feedback: {prompts}. Give detailed advice on how to improve it. Dont give more than 3 points and be as direct and straight to the point as possible.",
        max_tokens=300
    )
    return response.generations[0].text.strip()



def exercise_choose():
    exercise = " "

    if exercise.lower() == "squats":
        prompts = "While performing squats, "
        output = process_frame_for_squats() #func to squats
        if output[0]!=True:
            prompts += "heels are coming off ground, "
        if output[1] != True:
            prompts += "the angle that my shoulder hips and knees make is not correct, "
        if output[2] != True:
            prompts += "the angle that my hips, knee and ankle make is not correct "

    if exercise.lower() == "bench press":
        prompts = "While performing a bench press, "
        output = process_frame_for_bench_press() #func to bench press
        if output[0]!=True:
            prompts += "wrist and elbow dont create a 90 degree angle with eachother, "
        if output[1] != True:
            prompts += "the elbows flare out, "
        if output[2] != True:
            prompts += "the  bar isn't positioned near the sternum "

    if exercise.lower() == "dead lift":
        prompts = "While performing a dead lift, "
        output = process_frame_for_deadlift() #func to bench press
        if output[0]!=True:
            prompts += "knees goes past the toes, "
        if output[1] != True:
            prompts += "spine is curved, "
        if output[2] != True:
            prompts += "the angle that my shoulder hips and knees make is not correct, "
        if output[3] != True:
            prompts += "the angle that my hips, knee and ankle make is not correct "





    




