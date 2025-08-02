import cv2
import numpy as np
import os
from datetime import datetime
from insightface.app import FaceAnalysis

# ========== CẤU HÌNH ==========
DATASET_DIR = 'dataset'
SKIP_FRAMES = 5
THRESHOLD = 1.0  # Ngưỡng khoảng cách cosine để xác định đúng khuôn mặt

# ========== KHỞI TẠO MÔ HÌNH INSIGHTFACE ==========
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

# ========== LOAD KHUÔN MẶT MẪU ==========
def load_known_faces():
    known_faces = []
    for filename in os.listdir(DATASET_DIR):
        filepath = os.path.join(DATASET_DIR, filename)
        name, ext = os.path.splitext(filename)
        img = cv2.imread(filepath)
        faces = face_app.get(img)
        if faces:
            emb = faces[0].embedding
            known_faces.append((name, emb))
        else:
            print(f"[!] Không nhận diện được khuôn mặt trong {filename}")
    return known_faces

known_faces = load_known_faces()

# ========== KHOẢNG CÁCH COSINE ==========
def cosine_distance(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return 1 - np.dot(a, b)

# ========== GHI LOG ==========
def log_attendance(name):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("attendance_log.csv", "a") as f:
        f.write(f"{name},{now}\n")
    print(f"[+] Điểm danh: {name} lúc {now}")

attendance_set = set()

# ========== NHẬN DIỆN & VẼ KẾT QUẢ ==========
def recognize_and_draw(faces, frame):
    for face in faces:
        emb = face.embedding
        name = "Unknown"
        min_dist = float("inf")
        for known_name, known_emb in known_faces:
            dist = cosine_distance(emb, known_emb)
            if dist < THRESHOLD and dist < min_dist:
                name = known_name
                min_dist = dist

        box = face.bbox.astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(frame, name, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if name != "Unknown" and name not in attendance_set:
            attendance_set.add(name)
            log_attendance(name)

# ========== VÒNG LẶP CHÍNH ==========
cap = cv2.VideoCapture(0)
frame_count = 0
last_faces = []

print("[INFO] Bắt đầu điểm danh... Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Cập nhật nhận diện mỗi SKIP_FRAMES frame
    if frame_count % SKIP_FRAMES == 0:
        last_faces = face_app.get(frame)

    recognize_and_draw(last_faces, frame)

    cv2.imshow("Face Attendance", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

