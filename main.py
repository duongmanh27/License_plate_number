import os
import sqlite3
import threading
import queue
from tkinter import Label, Canvas, Scrollbar
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2 as cv
import numpy as np
import torch
from yolox.tracker.byte_tracker import BYTETracker
import easyocr
import tkinter as tk


db_path = "F:\\Code\\Python\\Project\\License_Plate_Number\\license_plate_2.db"
video_path = "F:\\Code\\Python\\Project\\License_Plate_Number\\data\\duong_thuong.mp4"
out_path = "F:\\Code\\Python\\Project\\License_Plate_Number\\outs_ne"

list_text = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
reader = easyocr.Reader(['en'])
yolov8s_pre = YOLO("F:\\Code\\Python\\Project\\License_Plate_Number\\yolov8n_pre_trained_lpn\\models\\best.pt")

os.makedirs(out_path, exist_ok=True)


def update_scroll_region(event) :
    canvas.configure(scrollregion=canvas.bbox("all"))


class Args :

    def __init__(self) :
        self.track_thresh = 0.5
        self.track_buffer = 50
        self.match_thresh = 0.4
        self.mot20 = False
        self.new_track_thresh = 0.6
        self.high_thresh = 0.6
        self.low_thresh = 0.2
        self.lost_frame_thresh = 100


conn = sqlite3.connect(db_path, check_same_thread=False)
cursor = conn.cursor()
cursor.execute(""
               "CREATE TABLE IF NOT EXISTS license_plates ( "
               "id INTEGER PRIMARY KEY AUTOINCREMENT, "
               "plate_text TEXT NOT NULL, "
               "confidence REAL NOT NULL, "
               "image_path TEXT NOT NULL "
               ");")
conn.commit()
root = tk.Tk()
root.title("License Plate Number App")
root.geometry("1300x900")

plates_frame = tk.Frame(root, width=200, height=720, bg="white")
plates_frame.pack(side="right", padx=10, pady=10, fill="y")

video_frame = tk.Frame(root, width=1080, height=720)
video_frame.pack(side="left", padx=10, pady=10)
video_frame.pack_propagate(False)
video_label = tk.Label(video_frame)
video_label.pack()


def on_mouse_scroll(event) :
    canvas.yview_scroll(int(-1 * (event.delta // 120)), "units")


canvas = Canvas(plates_frame, bg='yellow', width=180, height=700)
scrollbar = Scrollbar(plates_frame, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas, bg="lightblue")

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.bind("<MouseWheel>", on_mouse_scroll)
canvas.configure(yscrollcommand=scrollbar.set)
scrollbar.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)


def update_scrollregion() :
    canvas.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox("all"))


def update_plate_image() :
    cursor.execute("SELECT image_path, plate_text FROM license_plates ORDER BY id DESC LIMIT 10")
    rows = cursor.fetchall()
    for widget in scrollable_frame.winfo_children() :
        widget.destroy()
    for row in rows :
        image_path, plate_text = row
        image = Image.open(image_path).resize((150, 70))
        image_tk = ImageTk.PhotoImage(image)

        label_image = Label(scrollable_frame, image=image_tk, bg="white")
        label_image.image = image_tk
        label_image.pack(pady=5)

        label_text = Label(scrollable_frame, text=plate_text, font=("Arial", 13), bg="white")
        label_text.pack(pady=2)
    update_scrollregion()


def refine_plate_text(plate) :
    results = reader.readtext(plate, detail=1, allowlist=list_text, blocklist='0123456789-.,:')
    if not results :
        return "", 0.0
    best_result = max(results, key=lambda x : x[2])
    plate_text = best_result[1]
    ocr_confidence = best_result[2]

    refined_results = reader.readtext(plate, detail=1, allowlist=list_text, blocklist='0123456789-.,:')
    refined_texts = [res[1] for res in refined_results]
    refine_text = " ".join(refined_texts).replace("\n", "").strip()
    if refine_text and len(refine_text) > 2 :
        plate_text = refine_text
    return plate_text, ocr_confidence


def process_image(plate, scale_factor=3) :
    h, w = plate.shape[:2]
    image = cv.resize(plate, (w * scale_factor, h * scale_factor), interpolation=cv.INTER_CUBIC)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image


args = Args()
tracker = BYTETracker(args)
license_plate = {}
list_plate_text = []
frame_queue = queue.Queue(maxsize=1)
cap = cv.VideoCapture(video_path)


def process_video() :
    while cap.isOpened() :
        flag, frame = cap.read()
        if not flag :
            return
        results = yolov8s_pre(frame, imgsz=(frame.shape[0], frame.shape[1]))
        detections = [[*box.xyxy[0].tolist(), float(box.conf[0].tolist()), float(box.conf[0].tolist())]
                      for box in results[0].boxes]
        detection_np = np.array(detections, dtype=np.float32) if detections else np.empty((0, 6), dtype=np.float32)
        img_info = img_size = (frame.shape[0], frame.shape[1])
        tracks = tracker.update(torch.tensor(detection_np), img_info, img_size)
        for track in tracks :
            x1, y1, x2, y2 = map(int, track.tlbr)
            track_id = track.track_id

            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0] or x1 >= x2 or y1 >= y2 :
                continue
            plate_image = frame[y1 : y2, x1 : x2]
            plate_image = process_image(plate_image, 3)
            plate_text, ocr_confidence = refine_plate_text(plate_image)
            if ocr_confidence > 0.45 :
                if track_id not in license_plate or ocr_confidence > license_plate[track_id]["confidence"] :
                    license_plate[track_id] = {'text' : plate_text, 'confidence' : ocr_confidence}
                    if plate_text not in list_plate_text :
                        list_plate_text.append(plate_text)
                        if plate_text :
                            image_filename = os.path.join(out_path, "{}_{:.4f}.jpg".format(plate_text, ocr_confidence))
                            cv.imwrite(image_filename, plate_image)
                            cursor.execute(
                                "INSERT INTO license_plates (plate_text, confidence, image_path) VALUES (?, ?, ?)",
                                (plate_text, ocr_confidence, image_filename))
                            conn.commit()
                            root.after(10, update_plate_image)
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv.putText(frame, "{}|{}".format(track_id, plate_text), (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0),
                1)
        if not frame_queue.full() :
            frame_queue.put(frame)


def update_gui() :
    if not frame_queue.empty() :
        frame = frame_queue.get()
        frame = cv.resize(frame, (1080, 720))
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame_tk = ImageTk.PhotoImage(frame)
        video_label.configure(image=frame_tk)
        video_label.image = frame_tk
    root.after(10, update_gui)


video_thread = threading.Thread(target=process_video)
video_thread.start()

update_plate_image()
update_gui()
root.mainloop()

cap.release()
cv.destroyAllWindows()
conn.close()
