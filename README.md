# pFedCTP Target-City Fine-Tuning 改進研究

### 基於個人化聯邦學習的跨城市交通預測與低資源目標城市微調

本專案以 **pFedCTP（Personalized Federated Learning for Cross-City Traffic Prediction）** 為基礎，探討在 **Target City 僅有少量交通資料** 的情況下，如何改善模型最後的 Fine-tuning 階段，降低過擬合並提升泛化能力。

pFedCTP 的核心概念是透過 **Personalized Federated Learning（個人化聯邦學習）**，讓多個城市在不直接交換原始交通資料的前提下，共享可轉移的交通模式，再將學到的知識適應到資料量較少的目標城市。

本專案沒有重新設計整個 pFedCTP，而是針對最後的 **Target-City Fine-Tuning** 流程進行多種輕量化改進實驗。

---

## 📌 研究動機

交通預測模型通常需要大量歷史感測器資料才能取得良好表現，但新建置或資料量較少的城市往往無法提供足夠的訓練資料。

若直接將不同城市的資料集中到同一台伺服器訓練，可能產生資料隱私、資料治理及跨組織共享上的問題。

聯邦學習提供另一種方式：

```text
Source City A ─┐
               │
Source City B ─┼── Federated Training ── Shared Traffic Knowledge
               │
Source City C ─┘
                            │
                            ▼
                    Target City Model
                            │
                            ▼
                     Local Fine-Tuning
```

各城市保留自己的原始資料，只交換模型相關資訊，再將學習到的跨城市知識轉移至資料不足的 Target City。

但是，當 Target City 只有非常少量的資料時，如果直接對完整模型進行固定 Epoch 的 Fine-tuning，很容易快速過擬合。

因此本研究主要探討：

> **在 Target City 僅有少量資料的情況下，如何改善 pFedCTP 的 Fine-tuning 流程，使模型保留跨城市知識，同時能適應目標城市？**

---

## 📖 Baseline：pFedCTP

本專案主要參考：

**Personalized Federated Learning for Cross-City Traffic Prediction**  
Yu Zhang, Hua Lu, Ning Liu, Yonghui Xu, Qingzhong Li, Lizhen Cui  
IJCAI 2024, pp. 5526–5534  

論文連結：  
https://www.ijcai.org/proceedings/2024/0611

原始程式：  
https://github.com/ZYuSdu/pFedCTP

pFedCTP 使用一個 Spatio-Temporal Network（ST-Net）處理不同城市的空間與時間交通模式，並透過個人化聯邦學習將參數區分為較適合跨城市共享的部分，以及保留城市特性的 Private Parameters。

概念上 ST-Net 可分成：

```text
Traffic Sequence
      │
      ├── Spatial Structure (SS)
      │
      ├── Temporal Pattern (TP)
      │
      ├── Spatio-Temporal Integration (ST)
      │
      └── Predictor
              │
              ▼
      Multi-step Traffic Forecast
```

### 主要模組

| 模組 | 功能 |
|---|---|
| **SS** | 建模道路節點之間的空間關係 |
| **TP** | 學習交通流量中的時間模式 |
| **ST** | 整合 Spatial 與 Temporal Feature |
| **Predictor** | 輸出未來多個時間步的交通預測結果 |

---

# 🧪 本專案的改進方向

原始方法在 Target City 階段會進行模型 Fine-tuning。

當 Target City 只有少量資料時，本專案觀察到完整模型持續訓練可能造成 Overfitting，因此比較以下六種 Fine-tuning Strategy。

| 方法 | 說明 |
|---|---|
| **Original** | 原始完整模型 Fine-tuning，使用固定訓練流程 |
| **Early Stop** | 使用 Validation Loss / Metric 判斷何時停止訓練 |
| **Predictor-only** | 凍結大部分模型，只更新 Predictor |
| **Private-only** | 保留 Shared Knowledge，只更新城市相關 Private Parameters |
| **Staged** | 第一階段先訓練 Predictor，第二階段再加入 Private Parameters |
| **Reg+Val** | 加入 Parameter Regularization，並使用 Validation 選擇最佳模型 |

核心想法是：

```text
Federated Training 已經學到跨城市知識
                  │
                  ▼
        Target City 資料非常少
                  │
          ┌───────┴────────┐
          │                │
   Full Fine-Tuning    Selective Fine-Tuning
          │                │
     容易過擬合       保留 Shared Knowledge
                           │
                           ▼
                    Target Adaptation
```

---

# ⭐ Private-only Fine-Tuning

本次實驗中效果最好的方法為 **Private-only Fine-Tuning**。

其主要想法是：

```text
Shared Parameters
跨城市共同學習的知識
        │
        └── Freeze

Private Parameters
Target City 專屬特徵
        │
        └── Fine-Tune
                │
                ▼
        Target City Prediction
```

Target City 資料有限時，不重新更新所有已經透過 Federated Learning 學到的參數，而是主要調整與目標城市相關的 Private Parameters。

這樣可以降低模型因為少量 Target Data 而破壞原有跨城市知識的風險。

---

# 📊 實驗設定

本研究使用四個真實交通資料集：

- **METR-LA**
- **PEMS-BAY**
- **Shenzhen**
- **Chengdu**

主要實驗設定：

| 設定 | 數值 |
|---|---:|
| Federated Rounds | 90 |
| Batch Size | 32 |
| Target City Data | 3 Days |
| GCN Layers | 1 |
| Forecast Horizon | 1–6 Steps |
| Evaluation Metrics | MAE / RMSE |

Target City 的資料採時間順序切分，而不是隨機切分：

```text
Training       Validation       Test
  80%              10%          10%
──────────────► 時間順序 ──────────────►
```

這樣可以避免未來時間點的資料洩漏到訓練資料中。

---

# 📈 實驗結果

以下為不同 Target-City Fine-Tuning Strategy 的平均 RMSE：

| 方法 | Average RMSE | 相較 Original |
|---|---:|---|
| Original | 3.557 | Baseline |
| Early Stop | 3.078 | 改善 |
| Predictor-only | 3.082 | 改善 |
| **Private-only** | **3.020** | **最佳** |
| Staged | 3.048 | 改善 |
| Reg+Val | 3.055 | 改善 |

其中 **Private-only Fine-Tuning** 的 Average RMSE 最低：

```text
Original     = 3.557
Private-only = 3.020
```

改善幅度：

```text
(3.557 - 3.020) / 3.557 × 100%
≈ 15.1%
```

也就是說，在本次實驗設定下，**Private-only Fine-Tuning 的平均 RMSE 約比原始 Fine-tuning 改善 15.1%**。

---

# 🔍 實驗觀察

本專案得到幾個主要觀察：

1. **Target City 資料量很少時，Full Fine-Tuning 容易過擬合。**
2. **Early Stopping 是成本很低、但有效的改善方式。**
3. **不一定要更新所有參數才能適應 Target City。**
4. **保留 Federated Learning 已學到的 Shared Knowledge 對低資源 Target City 很重要。**
5. 在本次比較中，**Private-only Fine-Tuning 的 Average RMSE 最低。**

因此，本專案的重點不是重新設計 pFedCTP，而是證明：

> **僅改變 Target-City Fine-Tuning Strategy，也可能明顯改善低資源城市的交通預測效果。**

---

# 🧠 Privacy-Preserving 的部分

本專案中的 Privacy-Preserving 主要來自 **Federated Learning 架構**。

不同城市不需要直接將原始交通資料集中到同一伺服器，而是透過模型參數與跨城市知識進行協作學習。

需要特別說明的是，本專案主要研究：

- Federated Learning
- Personalized Federated Learning
- Cross-City Knowledge Transfer
- Target-City Fine-Tuning

並**沒有宣稱使用**：

- Differential Privacy
- Homomorphic Encryption
- Secure Multi-Party Computation

---

# 📂 Repository Structure

```text
Privacy-Preserving-Data-Analysis-Project/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── code/
│   ├── config.py
│   ├── run_STNet.py
│   ├── run_pFedCTP.py
│   ├── run_pFedCTP_ES.py
│   │
│   ├── clients/
│   │   ├── base_client.py
│   │   └── target_client_es.py
│   │
│   ├── models/
│   │   └── ST_Net.py
│   │
│   ├── trainers/
│   │   ├── fedbase.py
│   │   ├── pFedCTP.py
│   │   ├── pFedCTP-woF.py
│   │   ├── pFedCTP-Trans.py
│   │   └── pFedCTP_ES.py
│   │
│   ├── utils/
│   │   ├── data_utils.py
│   │   ├── data_utils_es.py
│   │   ├── early_stopping.py
│   │   └── metrics.py
│   │
│   └── dataset/
│       └── data_reader.py
│
├── docs/
│   └── experiment-summary.md
│
└── Final project/
    ├── final project.pptx
    └── report.pdf
```

---

# 🛠️ 本專案新增／修改的實驗程式

相較於原始 pFedCTP，本專案主要新增與 Target-City Fine-Tuning 實驗相關的程式：

```text
code/run_pFedCTP_ES.py
code/clients/target_client_es.py
code/trainers/pFedCTP_ES.py
code/utils/data_utils_es.py
code/utils/early_stopping.py
```

### `run_pFedCTP_ES.py`

Fine-tuning Strategy 的主要實驗入口，可切換不同 Target-City Fine-Tuning 模式。

### `target_client_es.py`

負責 Target City 的 Fine-tuning 流程，包括不同參數 Freeze / Unfreeze Strategy。

### `pFedCTP_ES.py`

整合 pFedCTP Federated Training 與改進後 Target Fine-tuning 流程。

### `data_utils_es.py`

處理 Target City Training / Validation / Test Data。

### `early_stopping.py`

提供 Validation-based Early Stopping 與最佳模型保存機制。

---

# ▶️ 執行方式

## 1. Clone Repository

```bash
git clone https://github.com/ching9026/Privacy-Preserving-Data-Analysis-Project.git
cd Privacy-Preserving-Data-Analysis-Project/code
```

## 2. 建立環境

建議使用 Python 3.8 以上版本。

Linux：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt
```

Windows PowerShell：

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ../requirements.txt
```

> PyTorch 與 PyTorch Geometric 建議依照實際 CUDA 版本選擇對應安裝版本。

---

# 📦 Dataset

由於原始交通資料集檔案較大，本 Repository **沒有直接上傳 `.npy` Dataset**。

資料來源可參考原始 pFedCTP Repository：

https://github.com/ZYuSdu/pFedCTP

以及其使用的交通資料來源。

下載後建議放置為：

```text
code/dataset/
├── metr-la/
│   ├── dataset.npy
│   └── matrix.npy
├── pems-bay/
│   ├── dataset.npy
│   └── matrix.npy
├── shenzhen/
│   ├── dataset.npy
│   └── matrix.npy
└── chengdu/
    ├── dataset.npy
    └── matrix.npy
```

`.gitignore` 已排除大型 Dataset，避免意外 commit 至 GitHub。

---

# 🚀 Fine-Tuning 實驗範例

例如使用 Shenzhen 作為 Target City：

```bash
python run_pFedCTP_ES.py \
  --algo=pFedCTP \
  --batch_size=32 \
  --target_city=shenzhen \
  --num_rounds=90 \
  --local_epochs=150 \
  --target_epochs=50 \
  --gcn_layers=1 \
  --ft_mode=private_only
```

Fine-tuning Mode 包含：

```text
full
early_stopping
regularized
predictor_only
private_only
staged_reg_es
```

---

# 🧰 技術與研究主題

| 類別 | 技術 |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch |
| Graph Learning | Graph Neural Networks / GCN |
| Federated Learning | Personalized Federated Learning |
| Time Series | Traffic Forecasting |
| Domain Adaptation | Cross-City Knowledge Transfer |
| Fine-Tuning | Early Stopping / Parameter Freezing / Regularization |
| Evaluation | MAE / RMSE |

---

# 📄 專題文件

完整專題內容另外整理於：

### 專題報告

```text
Final project/report.pdf
```

### 專題簡報

```text
Final project/final project.pptx
```

### 實驗摘要

```text
docs/experiment-summary.md
```

### Demo 影片

https://drive.google.com/file/d/1X7A4GpaYEIeLw-CixJqGPB7Ax5ccFZ2Z/view

---

# 🔮 Future Work

後續可進一步研究：

- 比較 Target City 只有 1 / 3 / 5 / 7 Days Data 時的效果
- 在不同 Target City 重複實驗
- 使用多個 Random Seeds 驗證結果穩定度
- 比較各 Forecast Horizon 的 MAE / RMSE
- Adaptive Layer Freezing
- Parameter-Efficient Fine-Tuning
- 更進一步的 Regularization Strategy
- Differential Privacy 與 Federated Learning 的結合

---

# 🙏 Acknowledgements

本專案建立於 pFedCTP 的研究與開源實作之上：

**Yu Zhang, Hua Lu, Ning Liu, Yonghui Xu, Qingzhong Li, Lizhen Cui**  
*Personalized Federated Learning for Cross-City Traffic Prediction*  
IJCAI 2024, 5526–5534.

Paper：  
https://www.ijcai.org/proceedings/2024/0611

Original Repository：  
https://github.com/ZYuSdu/pFedCTP

本專案主要針對 **低資源 Target City Fine-Tuning** 進行額外實驗與改進。