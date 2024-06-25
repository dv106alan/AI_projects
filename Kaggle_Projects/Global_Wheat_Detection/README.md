## 全球小麥偵測 (Global Wheat Detection)  
Data source：Etienne David, Ian Stavness, Maggie, Phil Culliton. (2020). Global Wheat Detection . Kaggle. https://kaggle.com/competitions/global-wheat-detection
  
使用影像分析來幫助辨識小麥-  
分析全球不同品種的小麥圖片，訓練一個可以辨識不同品種之小麥頭的模型，並加以標記  

### Model
此專案使用Pytorch建立模型，運用torchvision中的FasterRCNN模型做訓練。  
#### FasterRCNN_ResNet50: 使用FasterRCNN_ResNet50模型進行訓練  
指定預設權重COCO_V1，提取需要訓練之權重進行訓練  
使用ResNet50 backbone  
```
===========================================================================
Layer (type:depth-idx)                             Param #
===========================================================================
├─GeneralizedRCNNTransform: 1-1                    --
├─BackboneWithFPN: 1-2                             --
|    └─IntermediateLayerGetter: 2-1                --
|    |    └─Conv2d: 3-1                            (9,408)
|    |    └─FrozenBatchNorm2d: 3-2                 --
|    |    └─ReLU: 3-3                              --
|    |    └─MaxPool2d: 3-4                         --
|    |    └─Sequential: 3-5                        (212,992)
|    |    └─Sequential: 3-6                        1,212,416
|    |    └─Sequential: 3-7                        7,077,888
|    |    └─Sequential: 3-8                        14,942,208
|    └─FeaturePyramidNetwork: 2-2                  --
|    |    └─ModuleList: 3-9                        984,064
|    |    └─ModuleList: 3-10                       2,360,320
|    |    └─LastLevelMaxPool: 3-11                 --
├─RegionProposalNetwork: 1-3                       --
|    └─AnchorGenerator: 2-3                        --
|    └─RPNHead: 2-4                                --
|    |    └─Sequential: 3-12                       590,080
|    |    └─Conv2d: 3-13                           771
|    |    └─Conv2d: 3-14                           3,084
├─RoIHeads: 1-4                                    --
|    └─MultiScaleRoIAlign: 2-5                     --
|    └─TwoMLPHead: 2-6                             --
|    |    └─Linear: 3-15                           12,846,080
|    |    └─Linear: 3-16                           1,049,600
|    └─FastRCNNPredictor: 2-7                      --
|    |    └─Linear: 3-17                           2,050
|    |    └─Linear: 3-18                           8,200
===========================================================================
Total params: 41,299,161
Trainable params: 41,076,761
Non-trainable params: 222,400
===========================================================================
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






