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
全球小麥資料集包含張高分辨率 RGB 圖像和 190,000 個標記的小麥頭，這些小麥頭是從世界上多個國家收集的，處於不同生長階段，具有廣泛的基因型。  
- 資料內容  
  - 圖像類型：高解析度的 RGB 圖像。  
  - 標記：約 190,000 個小麥頭標記。  
  - 收集來源：來自世界各地多個國家，包括不同生長階段和基因型的小麥植物。  
  - 收集方式：基於戶外圖像的小麥頭檢測。  
- 訓練資料集  
  - 覆蓋區域：主要來自歐洲（法國、英國、瑞士）和北美（加拿大）的 3,000 多張圖像。  
  - 用途：用於開發通用解決方案，估計小麥穗的數量和大小。  
- 測試資料集  
  - 覆蓋區域：包括來自澳洲、日本和中國的約 1000 張圖像。  
  - 用途：用於評估模型在不同基因型、環境和觀察條件下的性能。  


### Execution  
- 建立Dataset資料集, 提取CSV資料後加入座標資料
  建立小麥資料集，將圖片,標記座標,及ID等資料加入資料集中
- 建立CNN模型
  損失函數使用自建函數Averager(累計損失並做平均)
  優化器使用隨機梯度下降SGD，學習率0.005, 動量0.9, 權重衰減0.0005
  使用albumentations函式庫建立transfrom函式，這裡只使用翻轉
- 建立Dataloader, 將Dataset輸出之資料進行組合, 將批次量設為16
- 取出資料並查看標注是否正確
  ![image]() 
- 將資料及模型置於GPU中執行
- 編輯訓練模型程序，訓練2回合
  將每張Image及Target輸入模型進行前向傳播，將其損失加總並傳入損失函數中
  將損失計算結果進行反向傳播，並使用優化器更新權重
  ```
  Iteration #50 train loss: 1.0150204827105964
  Iteration #100 train loss: 0.7746749724493439
  Iteration #150 train loss: 0.779301008184453
  Epoch #0 train loss: 0.8755133863481356
  Iteration #200 train loss: 0.8922283283979564
  Iteration #250 train loss: 0.8761128209147222
  Iteration #300 train loss: 0.7208498736836595
  Epoch #1 train loss: 0.8604924928866472
  ```
  訓練完成後可以看到損失值為0.86，測試將其繼續訓練也無法有效下降，故此為此參數設計之極限
- 將測試圖片資料載入，使用訓練好的模型參數進行預測，輸出結果如下
  可看出訓練損失集validation損失一起下降，並且有持續下降趨勢，顯示此模型持續進行訓練後可以有更好得表現。  
  ![image](https://raw.githubusercontent.com/dv106alan/AI_projects/main/Kaggle_Projects/Waste_Classification/png/resnet_output.png)  
  在test資料集中隨機抽取9張圖片進行預測，可看見有8張預測正確，1張不正確，其準確率為0.89，準確率明顯提升  
  ![image](https://raw.githubusercontent.com/dv106alan/AI_projects/main/Kaggle_Projects/Waste_Classification/png/resnet_test.png)

### Conclution  
一開始使用CNN模型訓練，效果並不好，於是將模型置換成更為複雜之模型並做訓練，準確率明顯提升(準確率0.67->0.88)。  
在此研判是因為資料來源種類眾多(30種)，並且每一種類之圖片特徵之差異也較大，故需要更複雜的模型進行學習。  
ResNet50需要較長的訓練時間，故此作遷移學習只訓練部分權重可節省訓練時間，也可達到準確率提升之效果。  






