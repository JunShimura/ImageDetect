import cv2
import time
import tkinter as tk
from tkinter import scrolledtext
from ultralytics import YOLO
import threading

# static global
# YOLOモデルをロード
model = YOLO("best.pt")  # トレーニング済みモデルを指定

# ゲーム設定
time_limit = 30  # 制限時間 (秒)
score = 0  # 初期スコア
stop_game = False  # ゲームを停止するフラグ

# GUIのセットアップ
root = tk.Tk()
root.title("Object Detection Game")
root.geometry("500x400")

# メッセージ表示用のTextウィジェット
message_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=20, width=60)
message_box.pack(pady=10)

def update_gui_message(message):
    """GUI上のメッセージボックスにテキストを追加"""
    message_box.insert(tk.END, message + "\n")
    message_box.see(tk.END)
    root.update()

def get_camera():
    """
    利用可能なカメラデバイスを順番に試し、接続可能なものを取得する。
    """
    max_cameras_to_try = 10  # 試すカメラの最大数
    for camera_index in range(max_cameras_to_try):
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            update_gui_message(f"カメラを取得しました: デバイスインデックス {camera_index}")
            return camera_index
        else:
            cap.release()
    update_gui_message("利用可能なカメラが見つかりませんでした。")
    return None

def detect_objects(frame):
    """YOLOモデルでオブジェクトを検出し、ラベルと信頼度を返す"""
    results = model(frame)
    predictions = results[0].boxes.data
    objects = []
    for pred in predictions:
        class_id = int(pred[5])  # クラスID
        confidence = float(pred[4])  # 信頼度
        bbox = pred[:4]  # バウンディングボックス（x1, y1, x2, y2）
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        objects.append((model.names[class_id], confidence, area))
    return objects

def calculate_score(objects):
    """検出結果に基づいてスコアを計算"""
    global score
    for label, confidence, area in objects:
        # 基本加点: オブジェクトが存在するだけで+10点
        base_score = 10
        # 面積加点: 面積が大きいほど高得点（最大+50点）
        area_score = min(area // 1000, 50)
        # 信頼度加点: 100%に近い場合、または50%に近い場合に加点（最大+20点）
        confidence_score = max(20 - abs(confidence - 1.0) * 100, 0) + max(20 - abs(confidence - 0.5) * 100, 0)

        # ラベルによる加点
        if label == "kinoko":
            label_score = 20
        elif label == "takenoko":
            label_score = 30
        else:
            label_score = 0

        # 合計得点
        total = base_score + area_score + confidence_score + label_score
        update_gui_message(f"{label} (信頼度: {confidence:.2f}, 面積: {area:.0f}) -> 加点: {total}")
        score += total

def game_loop():
    global stop_game, score
    camera_id = get_camera()
    if camera_id is None:
        return

    cap = cv2.VideoCapture(camera_id)
    start_time = time.time()
    update_gui_message("ゲーム開始！制限時間は30秒です。")

    while not stop_game:
        elapsed_time = time.time() - start_time
        if elapsed_time > time_limit:
            update_gui_message(f"ゲーム終了！最終スコア: {score}")
            break

        ret, frame = cap.read()
        if not ret:
            update_gui_message("カメラ映像を取得できませんでした。")
            break

        objects = detect_objects(frame)
        predictions = model(frame)
        annotated_frame = predictions[0].plot()

        calculate_score(objects)

        cv2.imshow("DXI stage2", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            update_gui_message("ゲームを中断しました。")
            stop_game = True
            break

        # 適切なタイミングでGUIを更新
        root.update()

    cap.release()
    cv2.destroyAllWindows()
    
# スタートボタンでゲーム開始
def start_game():
    threading.Thread(target=game_loop).start()

# ストップボタンでゲーム停止
def stop_game_action():
    global stop_game
    stop_game = True

# GUIボタン
start_button = tk.Button(root, text="Start Game", command=start_game)
start_button.pack(pady=10)
stop_button = tk.Button(root, text="Stop Game", command=stop_game_action)
stop_button.pack(pady=10)

# GUIのメインループ
root.mainloop()
