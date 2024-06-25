## 廢棄物分類模型 (Waste_Classification)  
Data source：Alistair King，www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification  
使用GPU T4 x2
  
開發用於可回收垃圾及生活垃圾分類的機器學習模型-  
分析不同廢棄物材料的視覺特徵，比較預設和預設情況下廢棄物分類演算法的性能  


### Model
此專案使用Pytorch建立模型，分別使用CNN及ResNet建立模型，並比較兩者差異。  
#### CNN模型結構: 使用兩層卷積層(3X3,3C)
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 32, 224, 224]             896
              ReLU-2         [-1, 32, 224, 224]               0
         MaxPool2d-3         [-1, 32, 112, 112]               0
            Conv2d-4         [-1, 64, 112, 112]          18,496
              ReLU-5         [-1, 64, 112, 112]               0
         MaxPool2d-6           [-1, 64, 56, 56]               0
            Linear-7                  [-1, 512]     102,760,960
              ReLU-8                  [-1, 512]               0
            Linear-9                   [-1, 30]          15,390
================================================================
Total params: 102,795,742
Trainable params: 102,795,742
Non-trainable params: 0
----------------------------------------------------------------
```
#### ResNet: 使用ResNet50模型進行遷移訓練
指定預設權重IMAGENET1K_V2，只訓練fc層之權重，其餘凍結
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 112, 112]           9,408
            Conv2d-2         [-1, 64, 112, 112]           9,408
       BatchNorm2d-3         [-1, 64, 112, 112]             128
              ReLU-4         [-1, 64, 112, 112]               0
         MaxPool2d-5           [-1, 64, 56, 56]               0
       BatchNorm2d-6         [-1, 64, 112, 112]             128
              ReLU-7         [-1, 64, 112, 112]               0
            Conv2d-8           [-1, 64, 56, 56]           4,096
         MaxPool2d-9           [-1, 64, 56, 56]               0
...
```

### Dataset  
該資料集包含 15,000 張圖像（每張 256x256 像素）的綜合集合，描繪了 30 個不同類別的各種可回收材料、一般廢物和家居用品。此資料集每個類別 500 張影像，
每個子類別 250 張影像。  
該資料集涵蓋廢棄物類別和項目，包括：  
- 塑膠：此類別包括塑膠水瓶、汽水瓶、清潔劑瓶、購物袋、垃圾袋、食物容器、一次性餐具、吸管和杯蓋的圖像。
- 紙張和紙板：此類別包括報紙、辦公用紙、雜誌、紙箱和紙板包裝的圖像。
- 玻璃：此類別包括玻璃製成的飲料瓶、食品罐和化妝品容器的圖像。
- 金屬：此類別包括鋁製汽水罐、鋁製食品罐、鋼製食品罐和氣霧罐的圖像。
- 有機垃圾：此類別包括食物垃圾的圖像，例如皮、蔬菜殘渣、蛋殼、咖啡渣和茶袋。
- 紡織品：此類別包括服裝和鞋子的圖像。

### Execution  

- 建立Dataset資料集
將資料來源以亂數洗牌，再將其區分為Train, Val, Test資料集
- 建立CNN模型, 損失函數使用CrossEntropy, 優化器選擇Adam(0.001), 及建立transfrom(224*224, 標準化)
- 建立Dataloader, 批次量為32
- 將資料及模型置於GPU中執行
- 編輯訓練模型程序，訓練5回合，並計算損失，並顯示結果
  ```
  Start Training!
  Epoch [1/5], Train Loss: 3.0650, Val Loss: 2.5112
  Epoch [2/5], Train Loss: 1.7162, Val Loss: 1.9072
  Epoch [3/5], Train Loss: 0.5564, Val Loss: 2.1190
  Epoch [4/5], Train Loss: 0.2452, Val Loss: 2.2983
  Epoch [5/5], Train Loss: 0.1487, Val Loss: 2.3650
  Training completed!
  ```
  訓練完成後可以看到Validation的損失未持續下降，判斷此模型太小可學習之權重太少，無法有效學習特徵
  ![image](https://raw.githubusercontent.com/dv106alan/AI_projects/main/Kaggle_Projects/Waste_Classification/png/cnn_output.png)  
  在test資料集中隨機抽取9張圖片進行預測，可看見有6張預測正確，3張不正確，其準確率為0.67  
  ![image](https://raw.githubusercontent.com/dv106alan/AI_projects/main/Kaggle_Projects/Waste_Classification/png/cnn_test.png)  
- 將模型置換為ResNet，這裡使用ResNet50(深度殘差網路)  
- 建立ResNet50模型, 損失函數使用CrossEntropy, 優化器選擇Adam(0.001), 及建立transfrom(224*224, 標準化, 隨機翻轉30度)  
- 將模型設為預設權重(IMAGENET1K_V2)，訓練fc層之權重，其餘權重凍結
- 將資料及模型置於GPU中執行
- 編輯訓練模型程序，訓練5回合，並計算損失，並顯示結果
  ```
  Start ResNet50 Training!
  Epoch [1/5], Train Loss: 1.1894, Val Loss: 0.9468
  Epoch [2/5], Train Loss: 0.9375, Val Loss: 0.8151
  Epoch [3/5], Train Loss: 0.8189, Val Loss: 0.7297
  Epoch [4/5], Train Loss: 0.7420, Val Loss: 0.7172
  Epoch [5/5], Train Loss: 0.6787, Val Loss: 0.6765
  Training completed!
  ```
  可看出訓練損失集validation損失一起下降，並且有持續下降趨勢，顯示此模型持續進行訓練後可以有更好得表現。  
  ![image](https://raw.githubusercontent.com/dv106alan/AI_projects/main/Kaggle_Projects/Waste_Classification/png/resnet_output.png)  
  在test資料集中隨機抽取9張圖片進行預測，可看見有8張預測正確，1張不正確，其準確率為0.89，準確率明顯提升  
  ![image](https://raw.githubusercontent.com/dv106alan/AI_projects/main/Kaggle_Projects/Waste_Classification/png/resnet_test.png)

### Conclution  
一開始使用CNN模型訓練，效果並不好，於是將模型置換成更為複雜之模型並做訓練，準確率明顯提升(準確率0.67->0.88)。  
在此研判是因為資料來源種類眾多(30種)，並且每一種類之圖片特徵之差異也較大，故需要更複雜的模型進行學習。  
ResNet50需要較長的訓練時間，故此作遷移學習只訓練部分權重可節省訓練時間，也可達到準確率提升之效果。  






