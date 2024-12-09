import cv2
import time
import torch
from ultralytics import YOLO

# モデルの読み込み
#model = YOLO("best.pt")
model = YOLO("yolov8n.pt")

# カメラの初期化
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not connected")
    exit()

# 画像サイズ設定
screen_width = 640
screen_height = 480

cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

# 表示するテキストと座標
text = input("文字を入力") #"Hello, OpenCV!"
position = (int(screen_width/2), int(screen_height/2))  # x=50, y=200の位置に表示

# フォントと設定
font = cv2.FONT_HERSHEY_SIMPLEX  # フォントの種類
font_scale = 2.5                 # フォントサイズ
color = (0, 0, 255)              # 色（B, G, R）-> 赤
thickness = 2                    # 線の太さ

# カウントとインデックス
c = 0
i = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    if i == 0:
        # 推論と注釈付きフレームの生成
        predictions = model(frame)
        annotated_frame = predictions[0].plot()

        # フレームにテキストを描画
        cv2.putText(annotated_frame, text, position, font, font_scale, color, thickness)

        # 表示
        cv2.imshow("DXI stage2", annotated_frame)

        # キー操作
        key = cv2.waitKey(1)
        if (key & 0xff) == ord("q"):  # 'q'キーで終了
            break
        elif (key & 0xff) == ord("c"):  # 'c'キーで画像保存
            c += 1
            fname = time.strftime("%Y%m%d%H%M%S") + ".jpg"
            cv2.imwrite(fname, annotated_frame)
            print(f"{c}:{fname} saved")

    i = (i + 1) % 30  # フレームカウンタ

cap.release()
cv2.destroyAllWindows()
