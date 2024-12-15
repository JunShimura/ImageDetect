import cv2
import time
from ultralytics import YOLO

class Colors:
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERCE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'

def print_colored(color, str):
    print(f"{color}{str}{Colors.RESET}")

import sys
print_colored(Colors.BLUE, f"Python executable:{sys.executable}" )
print_colored(Colors.GREEN, f"Python version:{sys.version}")

# static global
# YOLOモデルをロード
model = YOLO("best.pt")  # トレーニング済みモデルを指定

# ゲーム設定
time_limit = 30  # 制限時間 (秒)
score = 0  # 初期スコア
all_scores = []  # 全てのスコアを格納するリスト


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


def calculate_score(objects):
    """検出結果に基づいてスコアを計算し、結果を返す"""
    global score
    scores = []
    for label, confidence, area, bbox in objects:
        # 基本加点: オブジェクトが存在するだけで+10点
        base_score = 10
        # 面積加点: 面積が大きいほど高得点（最大+50点）
        area_score = int(min(area // 1000, 50))
        # 信頼度加点: 100%に近い場合、または50%に近い場合に加点（最大+20点）
        confidence_score = int(max(20 - abs(confidence - 1.0) * 100, 0) + max(20 - abs(confidence - 0.5) * 100, 0))

        # ラベルによる加点
        if label == "kinoko":
            label_score = 20
        elif label == "takenoko":
            label_score = -30
        else:
            label_score = 0

        # 合計得点
        total = base_score + area_score + confidence_score + label_score
        scores.append((label, confidence, area, total, bbox))

    return scores


def display_scores(scores, frame):
    """スコアを表示"""
    for label, confidence, area, total, bbox in scores:
        if total < 0:
            colorEscape = Colors.BG_RED
        elif total < 25:
            colorEscape = Colors.BG_YELLOW
        elif total < 50:
            colorEscape = Colors.BG_GREEN
        elif total < 75:
            colorEscape = Colors.BG_BLUE
        else:
            colorEscape = Colors.BG_CYAN
        print_colored(colorEscape,f"{label} (信頼度: {confidence:.2f}, 面積: {area:.0f}) -> 加点: {int(total)}")

        # バウンディングボックスとラベルをフレームに描画
        x1, y1, x2, y2 = map(int, bbox)
        color = (0, 255, 0) if total >= 50 else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} ({int(total)}pts)", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.waitKey(10)


def display_sorted_scores(scores, frame):
    """スコアを昇順でソートして表示"""
    sorted_scores = sorted(scores, key=lambda x: x[3])  # スコアでソート
    display_scores(sorted_scores, frame)


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
            print("ゲーム終了！")
            # 最終スコアを表示
            print("=== 最終結果 ===")
            final_total_score = sum(score[3] for score in all_scores)
            display_sorted_scores(all_scores, frame)
            print(f"最終スコアの合計: {final_total_score}")
            break

        # カメラからフレームを取得
        ret, frame = cap.read()
        if not ret:
            print("カメラ映像を取得できませんでした。")
            break

        # 5秒ごとにスコアを計算
        if time.time() - last_score_time >= 5:
            objects = detect_objects(frame)
            frame_scores = calculate_score(objects)
            all_scores.extend(frame_scores)

            # 算出したスコアとアノテーションを表示
            display_scores(frame_scores, frame)

            # 0.5秒間表示をキープ
            cv2.imshow("Game", frame)
            cv2.waitKey(500)

            # 5秒ごとの合計スコアを表示
            frame_total = sum(score[3] for score in frame_scores)
            print(f"5秒ごとの合計スコア: {frame_total}")

            last_score_time = time.time()

        # フレームを表示（オプション）
        cv2.imshow("Game", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("ゲームを中断しました。")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
