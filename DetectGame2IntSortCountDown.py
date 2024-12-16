import cv2
import time
from ultralytics import YOLO

import ColorEscape as ce # local unofficial

import sys
ce.print_colored(ce.Colors.BLUE, f"Python executable:{sys.executable}" )
ce.print_colored(ce.Colors.GREEN, f"Python version:{sys.version}")

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
    """スコアを表示し、バウンディングボックスを同期して描画"""
    for label, confidence, area, total, bbox in scores:
        if total < 0:
            colorEscape = ce.Colors.BG_RED
            box_color = (0, 0, 255)  # Red
        elif total < 25:
            colorEscape = ce.Colors.BG_YELLOW
            box_color = (0, 255, 255)  # Yellow
        elif total < 50:
            colorEscape = ce.Colors.BG_GREEN
            box_color = (0, 255, 0)  # Green
        elif total < 75:
            colorEscape = ce.Colors.BG_BLUE
            box_color = (255, 0, 0)  # Blue
        else:
            colorEscape = ce.Colors.BG_CYAN
            box_color = (255, 255, 0)  # Cyan

        # ラベルをコンソールに表示
        ce.print_colored(
            colorEscape,
            f"{label} (信頼度: {confidence:.2f}, 面積: {area:05.0f}) -> 加点: {total:03}"
        )

        # フレーム上に描画
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, f"{label} ({total:03}pts)", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        # フレームを表示
        cv2.imshow("Game", frame)
        cv2.waitKey(50)

def display_countdown(frame, countdown_time):
    """カウントダウンを画面中央に表示する"""
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2
    # temp_frame = frame.copy()
    cv2.putText(frame, str(countdown_time), (center_x - 50, center_y),
        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
    # cv2.imshow("Game", temp_frame) 

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
            break

        # カメラからフレームを取得
        ret, frame = cap.read()
        if not ret:
            print("カメラ映像を取得できませんでした。")
            break
        
        cd = 5-int(time.time() - last_score_time)
        
        # 5秒ごとにスコアを計算
        if cd<= 0:
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
        else:
            display_countdown(frame,cd)
        # フレームを表示（オプション）
        cv2.imshow("Game", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("ゲームを中断しました。")
            break

    # 終了
    cv2.waitKey(500)
    print("ゲーム終了！")
    # 最終スコアを表示
    cv2.waitKey(500)
    print("=== 最終結果 ===")
    final_total_score = sum(score[3] for score in all_scores)

    # 統計情報を計算
    confidences = [score[1] for score in all_scores]
    areas = [score[2] for score in all_scores]
    points = [score[3] for score in all_scores]
    object_count = len(all_scores)

    min_conf = min(confidences) if confidences else 0
    max_conf = max(confidences) if confidences else 0
    min_area = min(areas) if areas else 0
    max_area = max(areas) if areas else 0
    min_score = min(points) if points else 0
    max_score = max(points) if points else 0

    # 全スコアを表示
    display_sorted_scores(all_scores, frame)

    # 統計情報を表示
    cv2.waitKey(500)
    ce.print_colored(ce.Colors.BG_RED, f"最小信頼度: {min_conf:.2f}")
    cv2.waitKey(250)
    ce.print_colored(ce.Colors.BG_CYAN, f"最大信頼度: {max_conf:.2f}")
    cv2.waitKey(500)
    ce.print_colored(ce.Colors.BG_RED,f"最小面積: {min_area:05.0f}")
    cv2.waitKey(250)
    ce.print_colored(ce.Colors.BG_CYAN, f"最大面積: {max_area:05.0f}")
    cv2.waitKey(500)
    ce.print_colored(ce.Colors.BG_RED,f"最小加点: {min_score:03}")
    cv2.waitKey(250)
    ce.print_colored(ce.Colors.BG_CYAN, f"最大加点: {max_score:03}")
    cv2.waitKey(500)
    ce.print_colored(ce.Colors.BG_GREEN,f"認識したオブジェクト数: {object_count}")

    cv2.waitKey(500)
    ce.print_colored(ce.Colors.REVERCE, "最終スコアの合計:")
    ce.print_colored(ce.Colors.REVERCE, f"{final_total_score}")
    cv2.waitKey(500)
    
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
