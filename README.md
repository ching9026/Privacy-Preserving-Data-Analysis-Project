# Improving Target-City Fine-Tuning in pFedCTP

### Privacy-Preserving Cross-City Traffic Prediction with Personalized Federated Learning

This project studies how to improve **target-city fine-tuning** in **pFedCTP**, a personalized federated learning framework for cross-city traffic prediction.

The original pFedCTP framework transfers traffic knowledge from multiple data-rich source cities to a data-scarce target city while keeping each city's raw traffic data local. This project focuses on a practical issue that appears after federated training: when only a small amount of target-city data is available, fixed-epoch full-model fine-tuning can overfit quickly and reduce generalization.

The main goal is therefore to improve the target adaptation stage **without redesigning the entire pFedCTP framework**.

---

## Project Motivation

Traffic forecasting models usually benefit from large amounts of historical sensor data. However, newly deployed or less-developed cities may have only a small amount of local data.

A centralized cross-city training approach could combine data from several cities, but directly collecting all traffic data on one server creates privacy and governance concerns.

Federated learning provides an alternative:

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

pFedCTP addresses cross-city spatial and temporal heterogeneity with personalized federated learning. This project investigates whether the **final target-city fine-tuning stage** can be made more stable when only limited target data is available.

---

## Research Question

> How can target-city fine-tuning in pFedCTP reduce overfitting and improve generalization when the target city has only a few days of training data?

Instead of changing the complete federated architecture, this project explores lightweight fine-tuning strategies that can be added after federated training.

---

## Baseline: pFedCTP

The project is based on:

**Personalized Federated Learning for Cross-City Traffic Prediction**  
Yu Zhang, Hua Lu, Ning Liu, Yonghui Xu, Qingzhong Li, Lizhen Cui  
IJCAI 2024, pp. 5526–5534  
https://www.ijcai.org/proceedings/2024/0611

pFedCTP uses a spatio-temporal neural network (**ST-Net**) and personalized federated learning to transfer useful traffic patterns across cities while preserving inter-city data privacy.

Conceptually, ST-Net contains four major components:

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

The framework separates transferable traffic knowledge from city-specific information and then adapts the resulting model to a low-resource target city.

---

## Proposed Fine-Tuning Strategies

This project compares the original target-city fine-tuning procedure with several lightweight alternatives.

| Strategy | Description |
|---|---|
| **Original** | Original full-model target-city fine-tuning with a fixed training schedule |
| **Early Stop** | Use validation performance to stop fine-tuning before overfitting |
| **Predictor-only** | Fine-tune only the final prediction module |
| **Private-only** | Fine-tune only city-specific/private modules while keeping shared knowledge fixed |
| **Staged** | Warm up the predictor first, then fine-tune private modules together with the predictor |
| **Reg+Val** | Add parameter-distance regularization and validation-based model selection |

The key idea is to reduce unnecessary updates to already-learned cross-city knowledge when the target dataset is extremely small.

---

## Experimental Setting

The experiments use four real-world traffic datasets:

- **METR-LA**
- **PEMS-BAY**
- **Shenzhen**
- **Chengdu**

Important settings used in the fine-tuning study include:

| Setting | Value |
|---|---|
| Federated rounds | 90 |
| Batch size | 32 |
| Target-city data | 3 days |
| GCN layers | 1 |
| Data split | Chronological 80% / 10% / 10% |
| Forecast horizons | 1–6 steps |
| Metrics | MAE, RMSE |

A chronological split is used so validation and test samples occur later in time than the corresponding training samples.

---

## Main Results

The following table summarizes the average RMSE observed in the target-city fine-tuning experiments.

| Method | Average RMSE | Relative to Original |
|---|---:|---:|
| Original | 3.557 | Baseline |
| Early Stop | 3.078 | Improved |
| Predictor-only | 3.082 | Improved |
| **Private-only** | **3.020** | **Best** |
| Staged | 3.048 | Improved |
| Reg+Val | 3.055 | Improved |

Among the tested strategies, **Private-only fine-tuning achieved the lowest average RMSE (3.020)**, corresponding to an improvement of approximately **15.1%** over the original target-city fine-tuning result.

This suggests that, under a very small target-city dataset, updating only city-specific parameters can preserve useful cross-city knowledge while still allowing the model to adapt to the target domain.

---

## Key Takeaways

The experiments support three practical observations:

1. **Fixed-epoch target fine-tuning can overfit easily** when only a few days of target-city data are available.
2. **Validation-based early stopping is already a strong lightweight improvement** and requires little architectural change.
3. **Restricting which parameters are updated can outperform full-model fine-tuning**, with Private-only fine-tuning performing best in the reported experiments.

The project therefore demonstrates that target adaptation can be improved through training strategy changes rather than a full redesign of the federated learning framework.

---

## Repository Structure

```text
Privacy-Preserving-Data-Analysis-Project/
├── README.md
├── docs/
│   └── experiment-summary.md
└── Final project/
    ├── final project.pptx
    └── report.pdf
```

### Project Report

`Final project/report.pdf`

Contains the complete course project report, methodology discussion, experimental design, and analysis.

### Presentation

`Final project/final project.pptx`

Contains the final project presentation used to explain the motivation, pFedCTP framework, proposed fine-tuning strategies, and results.

### Experiment Summary

`docs/experiment-summary.md`

Provides a GitHub-readable summary of the experimental setup and results without requiring the PDF or PowerPoint to be downloaded.

---

## Tech / Research Topics

- Python
- PyTorch
- Federated Learning
- Personalized Federated Learning
- Graph Neural Networks
- Spatio-Temporal Modeling
- Traffic Forecasting
- Cross-City Knowledge Transfer
- Privacy-Preserving Machine Learning
- Target-Domain Fine-Tuning
- Early Stopping
- Parameter Regularization

---

## Privacy-Preserving Perspective

The privacy-preserving property in this project comes from the **federated learning setting**: participating cities collaboratively learn transferable model knowledge without directly centralizing their raw traffic datasets.

This project does **not** claim to implement differential privacy, homomorphic encryption, or secure multi-party computation. Its focus is personalized federated learning and low-resource cross-city adaptation.

---

## Reproducibility Note

This repository currently contains the **project report and presentation**, but not the full experimental source code or datasets.

The documentation is intended to preserve the research design, experimental comparison, and conclusions of the project. Reproducing the experiments requires the original pFedCTP implementation and the corresponding traffic datasets.

Original pFedCTP project/paper implementation details should be consulted together with the IJCAI 2024 paper.

---

## Possible Future Work

Potential extensions include:

- Add the modified pFedCTP source code used in the experiments
- Provide reproducible configuration files for each fine-tuning strategy
- Add per-horizon MAE / RMSE tables and plots
- Compare different target-data budgets, such as 1, 3, 5, and 7 days
- Study whether the best fine-tuning strategy changes across target cities
- Add repeated runs with multiple random seeds
- Explore adaptive layer freezing based on validation performance
- Evaluate stronger regularization or parameter-efficient adaptation methods

---

## Reference

Yu Zhang, Hua Lu, Ning Liu, Yonghui Xu, Qingzhong Li, and Lizhen Cui.  
**Personalized Federated Learning for Cross-City Traffic Prediction.**  
Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence (IJCAI-24), 5526–5534.

Official paper page:  
https://www.ijcai.org/proceedings/2024/0611
