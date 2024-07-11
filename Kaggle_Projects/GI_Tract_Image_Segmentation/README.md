## 水稻影像識別分類 (Rice Image Classification) (Pytorch)  
Data source：Rice Image Dataset, [https://kaggle.com/competitions/global-wheat-detection](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset)  

- **Core**: GPU T4 x2  

**此專案使用 CNN 模型對水稻品種進行分類。**  

- **資料集內容為：**  
  
  • 水稻圖片包含**Arborio、Basmati、Ipsala、Jasmine 和Karacadag** 五種稻米品種。  
  • 資料集(1) 有**75K 張影像**，其中每個水稻品種有15K 張影像。


    
**摘要**
<details open>
<summary>點擊展開說明</summary><br>

稻米是世界上生產最廣泛的糧食產品之一，有許多遺傳品種。這些品種由於它們的某些特徵而彼此分開。這些通常是紋理、形狀和顏色等特徵。利用這些區分稻米品種的特徵，可以對種子的品質進行分類和評估。在這項研究中，使用了土耳其經常種植的五種不同的稻米品種：Arborio、Basmati、Ipsala、Jasmine 和 Karacadag。資料集中包含總共 75,000 張穀物影像，其中每個品種 15,000 張。
  
</details>

**執行步驟說明：**  
    
- **資料處理**
    
  使用split-folders進行資料分類，此函式庫可以依照使用者設定，將資料分成**訓練、驗證及測試**資料集。  
  **訓練**樣本總數為**52500**，**驗證**總數為**11250**，**測試**總數為**11250**。  
    
- **建立資料集**  
    
  由於資料來源單純**無需修改**，所以這裡直接使用torchvision的ImageFolder功能建立Dataset。  
  Dataloader的部分也是直接調用Pytorch函式庫建立。  
  使用transforms更改樣本**圖片大小為(250,250)**，並作**標準化處理**。
    
  資料處理後如下：
  ```
  Train
  Shape of images [Batch_size, Channels, Height, Width]: torch.Size([32, 3, 250, 250])
  Shape of y: torch.Size([32]) torch.int64
  ---------------------------------------------
  validation
  Shape of images [Batch_size, Channels, Height, Width]: torch.Size([32, 3, 250, 250])
  Shape of y: torch.Size([32]) torch.int64
  ---------------------------------------------
  Test
  Shape of images [Batch_size, Channels, Height, Width]: torch.Size([32, 3, 250, 250])
  Shape of y: torch.Size([32]) torch.int64
  ---------------------------------------------
  ```
    
  **水稻圖片預覽**(顯示一批次，資料內容隨機排列)：  
    
  <img src="./imgs/rice_view.png" width="100%">  
    
  

- **建立模型**  
此專案使用 **CNN** 建立模型，此模型分為**卷積層及密集層**，  
卷積區塊有**三層CNN**，包含**卷積層、池化層級批次標準化**。  
密集層包含**三層線性層**，**輸出為5**個類別。  
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 32, 248, 248]             896
              ReLU-2         [-1, 32, 248, 248]               0
         MaxPool2d-3         [-1, 32, 124, 124]               0
       BatchNorm2d-4         [-1, 32, 124, 124]              64
            Conv2d-5         [-1, 64, 122, 122]          18,496
              ReLU-6         [-1, 64, 122, 122]               0
         MaxPool2d-7           [-1, 64, 61, 61]               0
       BatchNorm2d-8           [-1, 64, 61, 61]             128
            Conv2d-9          [-1, 128, 59, 59]          73,856
             ReLU-10          [-1, 128, 59, 59]               0
        MaxPool2d-11          [-1, 128, 29, 29]               0
      BatchNorm2d-12          [-1, 128, 29, 29]             256
           Linear-13                  [-1, 128]      13,779,072
             ReLU-14                  [-1, 128]               0
           Linear-15                   [-1, 64]           8,256
             ReLU-16                   [-1, 64]               0
           Linear-17                    [-1, 5]             325
================================================================
Total params: 13,881,349
Trainable params: 13,881,349
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.72
Forward/backward pass size (MB): 64.15
Params size (MB): 52.95
Estimated Total Size (MB): 117.82
----------------------------------------------------------------
```
    
- **訓練模型**  
    
  損失函數使用**交叉熵損失**函數，優化器使用**Adam**。  
  進行5次迭代訓練，每次訓練皆包含驗證，並統計訓練損失及正確率。  
  訓練完儲存模型參數。  

- **訓練結果**  
    
  經過五次迭代訓練結果如下：  
  可以看到訓練及驗證的**正確率都超過99%**  
  ```
  Epoch [1/5] -> Train Loss:0.0845, Train Acc:0.9740| Val Loss:0.0673, Val Acc:0.9791| Duration : 0:02:41.066440
  Epoch [2/5] -> Train Loss:0.0342, Train Acc:0.9889| Val Loss:0.0187, Val Acc:0.9944| Duration : 0:02:39.578081
  Epoch [3/5] -> Train Loss:0.0407, Train Acc:0.9874| Val Loss:0.0396, Val Acc:0.9865| Duration : 0:02:40.185162
  Epoch [4/5] -> Train Loss:0.0286, Train Acc:0.9911| Val Loss:0.0195, Val Acc:0.9943| Duration : 0:02:39.710852
  Epoch [5/5] -> Train Loss:0.0265, Train Acc:0.9924| Val Loss:0.0221, Val Acc:0.9934| Duration : 0:02:39.905354
  ```
    
  **訓練統計資料趨勢**：  
    
  <img src="./imgs/rice_acc_loss.png" width="100%">  
    
  訓練及驗證損失皆持續減少，準確率也有持續上升的趨勢，表示模型有良好的學習特徵。  
    
- **評估模型**  
    
  使用測試資料評估模型表現，結果如下：  
  **正確率有99.32%**  
  ```
  Loss:0.0206
  Accuracy:0.9932
  ```
    
  **顯示預測圖片**：
    
  可以看到這批次的預測資料皆正確！  
  <img src="./imgs/rice_predict.png" width="100%">  
    
  **畫出混淆矩陣(Confusion Matrix)**：  
    
  將資料以混淆矩陣做統計，可以很容易的看出各類別預測的統計結果。  
    
  <img src="./imgs/rice_cm.png" width="50%">    
    
  可以看到總共有69個預測錯誤，在99%的準確率下還是有不少失誤，  
  但因為測試樣本有一萬多筆，所以只佔總數一小部分。  
    
- **總結**  
      
  在混淆矩陣下，很容易可以看出哪些種類的預測較差，可以針對預測較差得種類進行加強訓練，  
  例如建立資料擴增功能，不僅可以將訓練資料量增加，也可避免多次訓練產生過度配適。    
    
    
  


