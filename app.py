# -*- coding: utf-8 -*-

from flask import Flask, render_template, request
from predict_from_microphone import predict_emotion_from_microphone, emotion_labels

app = Flask(__name__)

# Initialize chances
chances = 3

def calculate_score(selected_emotion, predicted_emotion, confidence):
    if selected_emotion == predicted_emotion:
        if confidence >= 80:
            return 10
        elif confidence >= 60:
            return 8
        elif confidence >= 40:
            return 6
        elif confidence >= 20:
            return 4
        else:
            return 2
    else:
        return 0
@app.route('/', methods=['GET', 'POST'])
def index():
    global chances
    if request.method == 'POST':
        if 'record' in request.form:
            selected_emotion_index = int(request.form['emotion'])
            selected_emotion = emotion_labels[selected_emotion_index]
            emotion_index, confidence, all_confidences = predict_emotion_from_microphone()

            # Tahmin edilen duygunun etiketini al
            predicted_emotion_label = emotion_labels[emotion_index]

            score = calculate_score(selected_emotion, predicted_emotion_label, confidence)

            if predicted_emotion_label != selected_emotion:
                chances -= 1

            if chances == 0:
                chances = 3  # Reset chances after using all

            return render_template('result.html', selected_emotion=selected_emotion, predicted_emotion=predicted_emotion_label, confidence=confidence, score=score, chances=chances, all_confidences=all_confidences)

    return render_template('index.html', chances=chances)


if __name__ == '__main__':
    app.run(debug=True, port=5002)
