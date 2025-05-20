import tkinter as tk
from tkinter import messagebox, ttk
import os
import threading
import pickle
import json
import shutil
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.drawing_styles import get_default_hand_landmarks_style
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_DIR = './data'
LABELS_FILE = './labels.json'
MODEL_FILE = './model.p'
PICKLE_FILE = './data.pickle'

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Translator")
        self.root.geometry("400x200")
        self.root.configure(bg="#f0f0f0")

        frame = tk.Frame(root, bg="#f0f0f0")
        frame.pack(pady=20)

        tk.Label(frame, text="Enter Sign Label:", bg="#f0f0f0").grid(row=0, column=0, sticky='w')
        self.label_entry = tk.Entry(frame, width=25)
        self.label_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Button(frame, text="Add Sign", command=self.add_sign, width=15, bg="#d1e7dd").grid(row=1, column=0, pady=5)
        tk.Button(frame, text="Delete Sign", command=self.delete_sign, width=15, bg="#f8d7da").grid(row=1, column=1, pady=5)
        tk.Button(frame, text="Retrain Model", command=self.train_model, width=15, bg="#fff3cd").grid(row=2, column=0, pady=5)
        tk.Button(frame, text="Start Translator", command=self.start_translator, width=15, bg="#cfe2ff").grid(row=2, column=1, pady=5)

        tk.Label(frame, text="Signs: ", bg="#f0f0f0").grid(row=3, column=0, sticky='w')
        self.sign_list_label = ttk.Combobox(frame, state='readonly', width=25)
        self.sign_list_label.grid(row = 3, column=1, pady=5)

        self.load_labels()
        self.update_sign_list()
        self.sign_list_label.current(0)

    def load_labels(self):
        if os.path.exists(LABELS_FILE):
            with open(LABELS_FILE, 'r') as f:
                self.labels_dict = json.load(f)
        else:
            self.labels_dict = {}

    def save_labels(self):
        with open(LABELS_FILE, 'w') as f:
            json.dump(self.labels_dict, f)

    def update_sign_list(self):
        labels_text = "\n".join(self.labels_dict.values())
        self.sign_list_label.config(values=labels_text.splitlines())

    def add_sign(self):
        label = self.label_entry.get().strip()
        if not label:
            messagebox.showerror("Error", "Enter a sign label.")
            return

        if label in self.labels_dict.values():
            messagebox.showinfo("Info", f"Sign '{label}' already exists.")
            return

        class_id = str(len(self.labels_dict))
        self.labels_dict[class_id] = label
        self.save_labels()
        self.update_sign_list()

        threading.Thread(target=self.collect_images, args=(class_id, label)).start()

    def delete_sign(self):
        label = self.label_entry.get().strip()
        class_id = None
        for k, v in self.labels_dict.items():
            if v == label:
                class_id = k
                break

        if class_id is None:
            messagebox.showerror("Error", f"Sign '{label}' not found.")
            return

        del self.labels_dict[class_id]
        self.save_labels()

        shutil.rmtree(os.path.join(DATA_DIR, class_id), ignore_errors=True)
        self.update_sign_list()

        messagebox.showinfo("Info", f"Sign '{label}' successfully deleted.")

    def collect_images(self, class_id, label):
        os.makedirs(os.path.join(DATA_DIR, class_id), exist_ok=True)
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            cv2.putText(frame, 'Press Space to collect images', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
            cv2.imshow('Capture', frame)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break

        for i in range(100):
            ret, frame = cap.read()
            cv2.imshow('Capture', frame)
            cv2.imwrite(os.path.join(DATA_DIR, class_id, f'{i}.jpg'), frame)
            cv2.waitKey(50)

        cap.release()
        cv2.destroyAllWindows()

        messagebox.showinfo("Info", f"Sign '{label}' successfully added.")

    def train_model(self):
        data, labels = [], []
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

        for dir_ in os.listdir(DATA_DIR):
            for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
                img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                if results.multi_hand_landmarks:
                    data_aux, x_, y_ = [], [], []
                    for hand_landmarks in results.multi_hand_landmarks:
                        for lm in hand_landmarks.landmark:
                            x_.append(lm.x)
                            y_.append(lm.y)

                        for lm in hand_landmarks.landmark:
                            data_aux.extend([lm.x - min(x_), lm.y - min(y_)])

                    data.append(data_aux)
                    labels.append(dir_)

        pickle.dump({'data': data, 'labels': labels}, open(PICKLE_FILE, 'wb'))

        X = np.asarray(data)
        y = np.asarray(labels)

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        model = RandomForestClassifier()
        model.fit(x_train, y_train)

        acc = accuracy_score(model.predict(x_test), y_test)
        pickle.dump({'model': model}, open(MODEL_FILE, 'wb'))
        messagebox.showinfo("Training Complete", f"Accuracy: {acc*100:.2f}%")

    def start_translator(self):
        threading.Thread(target=self._run_translator).start()

    def _run_translator(self):
        model = pickle.load(open(MODEL_FILE, 'rb'))['model']
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            cv2.putText(frame, 'Press Space to close', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, get_default_hand_landmarks_style(), DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2))

                data_aux, x_, y_ = [], [], []
                for lm in results.multi_hand_landmarks[0].landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in results.multi_hand_landmarks[0].landmark:
                    data_aux.extend([lm.x - min(x_), lm.y - min(y_)])

                prediction = model.predict([np.asarray(data_aux)])[0]
                predicted_label = self.labels_dict.get(str(prediction), '?')

                x1 = int(min(x_) * frame.shape[1])
                y1 = int(min(y_) * frame.shape[0])
                cv2.putText(frame, predicted_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            cv2.imshow('Translator', frame)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()
