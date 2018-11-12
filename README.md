simple_face_recognition
====

# Overview
自作データセットを用いた機械学習による顔認識を行います。
誰でもディレクトリ構造を整えるだけで簡単に使用できることを目指しました。


# Discription
以下のようなディレクトリ構造を持つデータセットを用意します。

original_data
	├─ man_0
	|	image_man_0_0.jpg
	|	image_man_0_1.jpg
	|	...
	├─ man_1
	|	image_man_1_0.jpg
	|	image_man_1_1.jpg
	|	...
	...
	|
	└─ man_N
		image_man_N_0.jpg
		image_man_N_1.jpg
		...

original_dataというディレクトリの中に認識対象の人物名のディレクトリを作成します。
それぞれの人物名のディレクトリの中にその人物の画像を配置します。

上記のようなデータセットを用意するだけ、顔認識を行うことができます。

# Requirement
開発言語はPythonです。
また、機械学習にはpytorch、画像処理にはOpenCVを使用しているので、これらのパッケージをインストールしてください。


# Development environmental
OS : Windows 10  
Python : Python 3.5.3 |Anaconda 4.4.0 (64-bit)  
GPU : None  

pytorch : 0.4.1  
torchvision : 0.2.1  
opencv : 3.4.1  


# Usage
extract_face_images.pyで自作データセットから顔認識用のデータセットを作成し、train_model.pyでそのデータセットを学習した顔認識モデルを作成します。  
各プログラムの詳細は以下のようになります。

## extract_face_images.py
自作のデータセットから顔のみを抽出し、機械学習用のデータセットに成形するプログラムです。
上記で説明したoriginal_dataから以下のようなtraining_dataを作成します。

training_data  
	├─ man_0  
	|	face_image_man_0_0.jpg  
	|	face_image_man_0_1.jpg  
	|	...  
	├─ man_1  
	|	face_image_man_1_0.jpg  
	|	face_image_man_1_1.jpg  
	|	...  
	...  
	|  
	└─ man_N  
		face_image_man_N_0.jpg  
		face_image_man_N_1.jpg  
		...  

顔を抽出する際にcascadeを指定する必要があります。これは各々保存パスが異なるので、適宜変更してください。

## train_model.py
training_dataを基に機械学習を行い分類器を作成します。  
ネットワークにはCNNを使用しています。今回はシンプルさを意識したので、ネットワーク構造は32×32のRGB画像を対象にしたLeNetにしました。


# Finally
初心者のためプログラム内にミスや改善点が多くあると思いますが、温かい目で見守っていただければ幸いです。

