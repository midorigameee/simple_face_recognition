"""
LOAD_DIR
    |-manA
        |-imageA_0
        |-imageA_1
        ...
    |-manB
        |-imageB_0
        |-imageB_1
        ...
    ...

SAVE_DIR
    |-manA
        |-face_imageA_0
        |-face_imageA_1
        ...
    |-manB
        |-face_imageB_0
        |-face_imageB_1
        ...
    ...

上記のようなディレクトリ構造を持つLOAD_DIRから顔写真を抽出する。
抽出された顔画像は上記のようなディレクトリ構造を持つSAVE_DIRに保存される
"""

import cv2
import os


FACE_SIZE = None    # None以外ならその大きさにリサイズする
LOAD_DIR = "original_data"
SAVE_DIR = "training_data"

# 分類器の指定
cascade_file = "C:\\workspace_py\\Anaconda3\\envs\\py35\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt2.xml"
cascade = cv2.CascadeClassifier(cascade_file)


def detect_maxsize_face(target_image):
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    face_list = cascade.detectMultiScale(target_gray, minSize=(50, 50))        

    # 検出した顔に印を付ける
    f_x = 0
    f_y = 0
    f_size = 0
    for (x, y, w, h) in face_list:
        color = (0, 0, 225)
        pen_w = 1

        # 一番大きい検出結果を顔とする
        if w > f_size:
            f_x = x
            f_y = y
            f_size = w

    return len(face_list), f_x, f_y, f_size


def extract_face(image, resize=None):
    f_num, f_x, f_y, f_size = detect_maxsize_face(image)
    face_image = None

    # 顔があったらその場所を抽出する
    if f_num > 0:
        face_image = image[f_y:f_y+f_size, f_x:f_x+f_size]

        # 顔画像をリサイズ
        if resize is not None:
            size = (resize, resize)
            face_image = cv2.resize(face_image, size)

    return face_image


def extract_target_multi(target_dir, save_dir):
    # 対象画像の読み込み
    class_list = os.listdir(target_dir)

    # 保存用のディレクトリ作成
    if os.path.exists(save_dir) is not True:
        os.mkdir(save_dir)

    # クラスごとに画像を読み込む
    for class_name in class_list:
        print("\n----- {} -----" .format(class_name))

        # クラスごとに保存用のディレクトリ作成
        class_save_dir = os.path.join(save_dir, class_name)
        if os.path.exists(class_save_dir) is not True:
            os.mkdir(class_save_dir)

        # class内の画像のファイル名をリスト化
        target_list = os.listdir(os.path.join(target_dir, class_name))

        for idx, target_name in enumerate(target_list):
            target_path = os.path.join(target_dir, class_name, target_name)
            target_data = cv2.imread(target_path, cv2.IMREAD_COLOR)
            target_face = extract_face(target_data)

            if target_face is None:
                print("[{}]{}:\n=> Faces are not found." .format(idx, target_name))
            else:
                print("[{}]{}:\n=> Save a face image." .format(idx, target_name))

                # 顔画像をリサイズ
                if FACE_SIZE is not None:
                    size = (FACE_SIZE, FACE_SIZE)
                    target_face = cv2.resize(target_face, size)

                # 顔画像を保存
                save_name = "face_image_" + class_name + str(idx) + ".jpg"
                save_path = os.path.join(class_save_dir, save_name)
                cv2.imwrite(save_path, target_face)


if __name__ == "__main__":
    extract_target_multi(LOAD_DIR, SAVE_DIR)