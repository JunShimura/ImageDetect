# ウィンドウ立ち上げ
#--------------------------------

# Tkinterモジュールのインポート
import tkinter

# ウィンドウ（フレーム）の作成
root = tkinter.Tk()

# ウィンドウの名前を設定
root.title("demo_Tkinter")

# ウィンドウの大きさを設定
root.geometry("400x400")

# イベントループ（TK上のイベントを捕捉し、適切な処理を呼び出すイベントディスパッチャ）
root.mainloop()