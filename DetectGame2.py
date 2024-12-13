import cv2
import time
from ultralytics import YOLO

import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)

# static global
# YOLOモデルをロード
model = YOLO("best.pt")  # トレーニング済みモデルを指定

# ゲーム設定
time_limit = 30  # 制限時間 (秒)
score = 0  # 初期スコア

def get_camera():
    """
    利用可能なカメラデバイスを順番に試し、接続可能なものを取得する。
    """
    max_cameras_to_try = 10  # 試すカメラの最大数
    for camera_index in range(max_cameras_to_try):
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"カメラを取得しました: デバイスインデックス {camera_index}")
            return camera_index
        else:
            cap.release()
    print("利用可能なカメラが見つかりませんでした。")
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
        objects.append((model.names[class_id], confidence, area, bbox))
    return objects

def calculate_score(objects, frame):
    """検出結果に基づいてスコアを計算し、アノテーションを描画"""
    global score
    for label, confidence, area, bbox in objects:
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
        print(f"{label} (信頼度: {confidence:.2f}, 面積: {area:.0f}) -> 加点: {total}")
        score += int(total)

        # バウンディングボックスの色をスコアに応じて設定
        if total > 50:
            color = (0, 0, 255)  # 赤 (高得点)
        else:
            color = (0, 255, 0)  # 緑 (低得点)

        # バウンディングボックスとラベルをフレームに描画
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} ({int(total)}pts)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def main():
    # カメラの取得を試みる
    camera_id = get_camera()
    if camera_id is None:
        exit()

    # ゲーム開始
    cap = cv2.VideoCapture(camera_id)  # カメラを起動
    start_time = time.time()
    last_score_time = start_time

    print("ゲーム開始！制限時間は30秒です。")

    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > time_limit:
            print(f"ゲーム終了！最終スコア: {score}")
            break

        # カメラからフレームを取得
        ret, frame = cap.read()
        if not ret:
            print("カメラ映像を取得できませんでした。")
            break

        # 5秒ごとにスコアを計算
        if time.time() - last_score_time >= 5:
            objects = detect_objects(frame)
            calculate_score(objects, frame)
            last_score_time = time.time()

            # 採点結果を表示するため0.5秒停止
            cv2.imshow("Game", frame)
            cv2.waitKey(1000)

        # フレームを表示（オプション）
        cv2.imshow("Game", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("ゲームを中断しました。")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
