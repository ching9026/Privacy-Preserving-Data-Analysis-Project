from utils.data_utils import get_data, generate_dataset, traffic_dataset1


def getTargetTrainValTestDataSet_ES(options, city_name, target_days=3):
    """
    Early Stopping 版本的 target city data split.

    原本作者版本：
        target train: 前 target_days 天
        target test : 整份資料最後 20%

    這個版本：
        target sparse data = 前 target_days 天
        sparse data 再照時間順序切成 train / val / test

    Default:
        train: 60%
        val:   20%
        test:  20%
    """

    target_days = int(options.get("target_days", target_days))

    train_ratio = float(options.get("target_train_ratio", 0.6))
    val_ratio = float(options.get("target_val_ratio", 0.2))
    test_ratio = float(options.get("target_test_ratio", 0.2))

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            "target_train_ratio + target_val_ratio + target_test_ratio must be 1.0"
        )

    X, A_list, str_init, edge_index, means, stds = get_data(options, city_name)

    # 每天 288 個 time steps，作者原本也是用 288 * target_days
    sparse_steps = 288 * target_days
    X_sparse = X[:, :, :sparse_steps]

    x_all, y_all = generate_dataset(
        X_sparse,
        options["his_num"],
        options["pred_num"],
        means,
        stds,
    )

    total_samples = x_all.shape[0]

    n_train = int(total_samples * train_ratio)
    n_val = int(total_samples * val_ratio)
    n_test = total_samples - n_train - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"Invalid split. total_samples={total_samples}, "
            f"train={n_train}, val={n_val}, test={n_test}. "
            f"Please increase target_days or adjust split ratios."
        )

    x_train = x_all[:n_train]
    y_train = y_all[:n_train]

    x_val = x_all[n_train:n_train + n_val]
    y_val = y_all[n_train:n_train + n_val]

    x_test = x_all[n_train + n_val:]
    y_test = y_all[n_train + n_val:]

    train_dataset = traffic_dataset1(
        options,
        city_name,
        str_init,
        edge_index,
        A_list,
        x_train,
        y_train,
    )

    val_dataset = traffic_dataset1(
        options,
        city_name,
        str_init,
        edge_index,
        A_list,
        x_val,
        y_val,
    )

    test_dataset = traffic_dataset1(
        options,
        city_name,
        str_init,
        edge_index,
        A_list,
        x_test,
        y_test,
    )

    print("[Target Split for Early Stopping]")
    print(f"  city: {city_name}")
    print(f"  target_days: {target_days}")
    print(f"  total sparse samples: {total_samples}")
    print(f"  train samples: {len(train_dataset)}")
    print(f"  val samples:   {len(val_dataset)}")
    print(f"  test samples:  {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset