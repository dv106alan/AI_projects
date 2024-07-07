## 肺結節分析 (Lung Nodule Analysis)  
Reference：Deep Learning with PyTorch, Eli Stevens, Luca Antiga, and Thomas Viehmann  
Data Source：https://luna16.grand-challenge.org/  
此專案是參照Deep Learning with PyTorch之LUNA分析專案，將此專案實做於Kaggle平台上，並做模型訓練及分析。

#### 專案內容
此專案主要目的是要從CT掃描圖片中，辨識出肺結節並加以分析是否為癌化結節。  

本專案將作業切分為四個部分，分別為：  
- **[肺結節分類](./README.md)**	-         建立可處理3D-CT影像之CNN模型，並載入結節資料加以訓練。  
- **[結節偵測並做影像切割](./README.md)** -  建立Unet影像分割模型，將資料處理為2D批次圖像作為輸入資料格式，並載入結節資料訓練模型。  
- **[惡性結節分類](./README.md)** -       使用結節分類模型並作遷移訓練，載入惡性結節資料並做訓練。  
- **[整合模型並分析](./README.md)** -       整合上述3個模型，建立CT掃描程序，並加以分析。  

#### LUNA Grand Challenge 簡介

<details>
<summary>點擊展開說明</summary>
  
肺癌是全球癌症相關死亡的主要原因。美國的國家肺癌篩查試驗（NLST）顯示，對於高風險人群，使用年度低劑量電腦斷層掃描（CT）進行肺癌篩查比使用年度胸部X光篩查能降低20%的肺癌死亡率。2013年，美國預防服務工作組（USPSTF）對高風險人群的低劑量CT篩查給予B級推薦，2015年初，美國醫療保險和醫療補助服務中心（CMS）批准了對醫療保險受益者的CT肺癌篩查。隨著這些發展，使用低劑量CT的肺癌篩查計劃正在美國和其他國家實施。當篩查大規模實施時，計算機輔助檢測（CAD）對肺結節的檢測可能發揮重要作用。  
  
大型評估研究對不同先進CAD(Computer-aided detection)系統的性能進行調查較少。因此，我們使用大型公開的LIDC-IDRI數據集組織了一個新穎的CAD檢測挑戰。挑戰的詳細描述現在已在本文中提供。我們認為這個挑戰對於可靠地比較CAD算法以及鼓勵使用先進計算機視覺技術開發新算法具有重要意義。  

</details>

## 程式說明 - 肺結節分類

程式項目內容：
<details>
<summary>程式項目內容說明</summary>

- **公用程式**
  - Install Libraries - 安裝必要套件
  - Utils - 格式轉換、Log、訓練輔助程式
  - Disk - 快取存取程式
  - Visialize - 圖片顯示程式

- **資料集**
  - Datasets - 建立資料集程式
  - Nodule Sample - 範例顯示

- **模型**
  - Model - 建立模型程式

- **訓練程式**
  - Training - 建立訓練程式
  - Prepare Catch - 資料載入快取程式

- **訓練結果**
  - Start Training - 開始訓練及訓練結果

</details>

  










<!--
隱藏的文字：以下是各個功能模塊的詳細信息。
-->

## Features

<details>
  <summary>點擊展開詳細功能說明</summary>


| 公用程式 | 資料集 | 模型 | 訓練程式 | 訓練結果 |
|----------|--------|------|----------|----------|
| Install Libraries | Datasets | Model | Training | Start Training |
| Utils | Nodule Sample | | Prepare Catch | |
| Disk | | | | |
| Visualize | | | | |

