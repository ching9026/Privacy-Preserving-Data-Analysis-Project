# Experiment Summary

## Objective

This experiment studies how to make the **target-city adaptation stage of pFedCTP** more robust when the target city has only a small amount of local traffic data.

The baseline performs full-model fine-tuning for a fixed number of epochs. Under a low-resource setting, this can overfit quickly. The experiment therefore compares several lightweight alternatives that restrict updates, use validation feedback, or regularize the adaptation process.

---

## Compared Strategies

### Original

Fine-tune the target model using the original full-model procedure and fixed training schedule.

### Early Stop

Use a validation split to monitor target-city performance and stop training when validation error no longer improves.

### Predictor-only

Freeze most of the network and update only the final prediction module.

### Private-only

Keep shared cross-city knowledge fixed and update only the city-specific/private parts of the model.

### Staged

Use a multi-stage schedule:

```text
Stage 1
Predictor-only warm-up
        │
        ▼
Stage 2
Private modules + Predictor
```

The staged strategy aims to stabilize the prediction head before allowing broader target-specific adaptation.

### Reg+Val

Combine validation-based model selection with parameter-distance regularization so the target model does not drift too far from the pretrained/federated parameters.

---

## Experimental Configuration

| Parameter | Setting |
|---|---|
| Source/target traffic datasets | METR-LA, PEMS-BAY, Shenzhen, Chengdu |
| Federated rounds | 90 |
| Target-city data budget | 3 days |
| Batch size | 32 |
| GCN layers | 1 |
| Split | Chronological 80% train / 10% validation / 10% test |
| Prediction horizons | Steps 1–6 |
| Metrics | MAE, RMSE |

The chronological split is important for traffic forecasting because random splitting may leak future temporal patterns into the training set.

---

## Average RMSE Results

| Rank | Method | Average RMSE |
|---:|---|---:|
| 1 | **Private-only** | **3.020** |
| 2 | Staged | 3.048 |
| 3 | Reg+Val | 3.055 |
| 4 | Early Stop | 3.078 |
| 5 | Predictor-only | 3.082 |
| 6 | Original | 3.557 |

### Improvement of Best Strategy

Using the original result as the baseline:

```text
(3.557 - 3.020) / 3.557 × 100%
≈ 15.1%
```

Therefore, **Private-only fine-tuning reduced average RMSE by approximately 15.1%** in the reported experiment.

---

## Interpretation

The results indicate that the federated model already contains useful transferable traffic knowledge. When the target dataset is very small, updating the complete model can destroy part of that knowledge or over-specialize the network to the limited local samples.

Private-only fine-tuning provides a useful compromise:

```text
Federated Shared Knowledge
        │
        ├── keep fixed
        │
        ▼
Target-specific / Private Parameters
        │
        ├── fine-tune
        │
        ▼
Target-city Prediction
```

This allows target adaptation while reducing unnecessary changes to the transferable representation learned across source cities.

---

## Conclusions

The experiment suggests that improving **how** the target model is fine-tuned can be more effective than simply increasing the number of fine-tuning epochs.

The main findings are:

- full-model fixed-epoch fine-tuning is vulnerable to overfitting under limited target data;
- validation-based early stopping provides a simple improvement;
- parameter freezing can improve generalization;
- Private-only fine-tuning achieved the best average RMSE among the tested strategies;
- lightweight target adaptation can improve pFedCTP without modifying the overall federated learning architecture.

---

## Suggested Follow-up Experiments

A stronger evaluation could investigate:

1. different target-data budgets (1 / 3 / 5 / 7 days);
2. multiple target cities;
3. several random seeds;
4. per-horizon MAE and RMSE instead of only average metrics;
5. sensitivity to regularization strength;
6. adaptive layer freezing;
7. parameter-efficient fine-tuning methods.
