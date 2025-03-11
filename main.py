from flask import Flask, render_template, Response, request, jsonify
import cv2
import time
import mediapipe as mp
from ultralytics import YOLO
import kenlm
import nltk
from nltk.corpus import words, wordnet

app = Flask(__name__)

# Initialize the webcam variable as None, which means the webcam is off initially
cap = None

model_path = f"./static/models/YOLO.pt"
model = YOLO(model_path).to("cpu")                                                        #
class_names = model.model.names

# Download language corpora
nltk.download("words")  # english
# italian
with open("./static/corpora/280000_parole_italiane.txt", "r") as file:
    italian_words = file.read().split("\n")
# spanish
with open("./static/corpora/spanish_words.txt", "r", encoding='latin-1') as file:
    spanish_words = file.read().split("\n")


# Define language vocabularies
ita_voc = set(italian_words)  # 279,895 words
eng_voc = set(words.words())  # 235,892 words
spa_voc = set(spanish_words)  # 76,441 words

# Set current language
current_language = 'english'

# Dictionary to map language to Hugging Face models
MODELS = {
    'english': {
        'model_name': 'kenLM_eng.binary',
        'vocabulary': eng_voc,
    },
    'italian': {
        'model_name': 'kenLM_ita.bin',
        'vocabulary': ita_voc,
    },
    'spanish': {
        'model_name': 'kenLM_es.binary',
        'vocabulary': spa_voc,
    }
}

# Get kenlm models
# Global dictionary to store loaded KenLM models
kenlm_models = {}


# Load models at startup
def load_kenlm_models():
    for lang, data in MODELS.items():
        model_name = data['model_name']
        if model_name:
            kenlm_models[lang] = kenlm.Model(f"./static/models/{model_name}")


load_kenlm_models()

# Variables to determine whether to show bboxes and hand landmarks
show_landmarks = False
show_bbox = False

# Initialize mediapipe Hands for bbox detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize variables
current_letter = None
frame_count = 0
required_stable_frames = 15  # stability parameter (must appear for 15 frames)
confidence = 0.70  # confidence threshold
word_list = []  # list to store recognized text
current_recognized_text = "Waiting for input..."
suggested_words = []
any_character = False  # check if any character has been detected to activate speaker button


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/set_language', methods=['POST'])
def set_language():
    global current_language
    data = request.get_json()
    if 'language' in data:
        current_language = data['language']
        print(f"Language set to: {current_language}")
        return jsonify(success=True)
    else:
        return jsonify(success=False), 400


# Start the webcam feed
@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    global cap
    if cap is not None:  # ensure the previous capture is fully released
        cap.release()
        cap = None
        time.sleep(0.5)  # small delay to allow the webcam to fully reset

    cap = cv2.VideoCapture(0)  # restart the webcam
    if not cap.isOpened():
        return "Error: Could not open webcam.", 500

    print("Webcam started!")
    return "Webcam started", 200


# Stop the webcam feed
@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    global cap
    if cap is not None:  # stop webcam only if it is running
        cap.release()
        cap = None
        time.sleep(0.5)

    # Clear recognized text
    clear_text()
    print("Webcam stopped!")
    return "Webcam stopped", 200


# Video feed streaming route
def generate_frames():
    global cap

    # Ensure cap is open before proceeding
    if cap is None or not cap.isOpened():
        return

    # Get the original frame width and height from the webcam
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate aspect ratio
    aspect_ratio = original_width / original_height

    # Set target width and calculate height dynamically
    target_width = 640
    target_height = int(target_width / aspect_ratio)  # this way we keep the aspect ratio

    while cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize while maintaining aspect ratio
        frame = cv2.resize(frame, (target_width, target_height))

        # Process frame
        frame, recognized_text = process_frame(frame)

        global current_recognized_text
        current_recognized_text = recognized_text

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


# Function to process each frame
def process_frame(frame):
    global current_letter, frame_count, word_list, suggested_words, any_character

    # Flip the frame for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert to RGB for mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands using MediaPipe
    results = hands.process(rgb_frame)

    # Process detected hands
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        hand_label = results.multi_handedness[0].classification[0].label

        # Get hand bounding box coordinates
        x_min, y_min, x_max, y_max = 1e6, 1e6, 0, 0
        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
            x_min, y_min = min(x_min, x), min(y_min, y)
            x_max, y_max = max(x_max, x), max(y_max, y)

        # Expand bounding box for better alignment with training set
        padding = 50
        x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
        x_max, y_max = min(frame.shape[1], x_max + padding), min(frame.shape[0], y_max + padding)

        # Crop the hand region
        hand_crop = frame[y_min:y_max, x_min:x_max]

        # Flip the image if it's a right hand (to match left-hand dataset)
        if hand_label == "Right":
            hand_crop = cv2.flip(hand_crop, 1)

        # Ensure valid crop before passing to YOLO
        if hand_crop.shape[0] > 0 and hand_crop.shape[1] > 0:
            # Run YOLO on the cropped hand image
            results_yolo = model(hand_crop, verbose=False)

            idx = results_yolo[0].probs.top1
            conf = results_yolo[0].probs.top1conf.item()
            label = class_names[idx]

            # Stability check: only update if the letter remains for required frames
            if label == current_letter:
                frame_count += 1
            else:
                frame_count = 0  # reset if letter changes
                current_letter = label  # update the current detected letter

            # If stable for required frames, add it to word list
            if frame_count >= required_stable_frames and conf >= confidence:
                if label == "space":
                    word_list.append("_")
                elif label == "del" and word_list:
                    word_list.pop()  # remove the last letter
                elif label not in ["space", "del", "nothing"]:
                    word_list.append(label)  # append detected letter
                frame_count = 0  # reset frame count after adding

                # Get suggested words
                suggested_words = predict_next_words(word_list) or []
                # print("Suggested words after detection: ")
                # print(suggested_words)
                # next_word_pred = predict_next_word(word_list)

                # Set that any character has been detected to activate speaker button
                any_character = True

            # Draw bounding box around hand region
            if show_bbox:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Draw classification label
            text = f"{label}: {conf:.2f}"
            cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Draw hand landmarks
        if show_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # print(word_list)
    # print(current_recognized_text)
    return frame, "".join(word_list).replace("_", " ").lower()


@app.route('/update_flags', methods=['POST'])
def update_flags():
    global show_landmarks, show_bbox

    data = request.get_json()
    if data['type'] == 'landmarks':
        show_landmarks = data['value']
    elif data['type'] == 'boxes':
        show_bbox = data['value']

    print(f"Updated: show_landmarks={show_landmarks}, show_boxes={show_bbox}")
    return jsonify(success=True)


def predict_next_words(wl):
    global current_language
    context = "".join(wl)
    splits = context.split("_")

    if not splits:
        return None  #

    # Extract previous words and the incomplete word
    if len(splits) == 1:
        previous_words, current_word = "", splits[0]
    else:
        previous_words, current_word = " ".join(splits[:-1]), splits[-1]

    # Filter vocabulary to only words that match the current prefix
    vocab = MODELS[current_language]['vocabulary']
    candidate_words = [w for w in vocab if w.lower().startswith(current_word.lower())]

    scored_words = []

    # Score each candidate word using KenLM
    model_lm = kenlm_models[current_language]  # get the preloaded model
    for word in candidate_words:
        test_sentence = f"{previous_words.lower()} {word}"  # keep previous context
        score = model_lm.score(test_sentence)
        scored_words.append((word, score))

    # Sort by score in descending order and get the top 3 suggestions
    top_suggestions = [word for word, _ in sorted(scored_words, key=lambda x: x[1], reverse=True)[:3]]

    return top_suggestions


def update_word_list(wl, word):
    # Find the index of the last underscore
    if '_' in wl:
        last_underscore_index = len(wl) - 1 - wl[::-1].index('_')
        # Keep everything up to and including the last underscore
        wl = wl[:last_underscore_index + 1]
    else:
        # If no underscore is found, clear the list
        wl = []

    # Append characters of the new word and a space
    wl.extend(word+'_')

    return wl


# Add this route to retrieve the current recognized text
@app.route('/get_recognized_text')
def get_recognized_text():
    global current_recognized_text
    return current_recognized_text or "Waiting for input..."


@app.route('/get_suggested_words')
def get_suggested_words():
    global suggested_words
    return jsonify(suggested_words if suggested_words else [])


@app.route('/select_suggested_word', methods=['POST'])
def select_suggested_word():
    global word_list, suggested_words
    idx = int(request.args.get('index', -1))
    # print(idx)

    if 0 <= idx < 3:
        if suggested_words and idx < len(suggested_words):
            word_list = update_word_list(word_list, suggested_words[idx])
            # Once a word is selected, immediately display next predicted words
            suggested_words = predict_next_words(word_list)             ## TODO: cancel here if it lags too much
        # print("Selected word: ", suggested_words[idx])
    return "Word selected", 200


@app.route('/any_character_detected', methods=['GET'])
def any_character_detected():
    global any_character
    return {"any_character_detected": any_character}


# Video feed route (serves MJPEG stream)
@app.route('/video_feed')
def video_feed():
    if cap is None:
        return "Webcam is not started yet", 400
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/clear_text', methods=['POST'])
def clear_text():
    global word_list, current_recognized_text, any_character

    # Clear the list and reset the text
    word_list = []
    current_recognized_text = "Waiting for input..."
    any_character = False

    print("Text cleared!")
    return "Text cleared", 200


if __name__ == '__main__':
    app.run(debug=True)
