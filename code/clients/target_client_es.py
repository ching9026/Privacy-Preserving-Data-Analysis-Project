import torch
from torch import optim

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader

from clients.base_client import BaseClient
from utils.metrics import metric_func
from utils.early_stopping import EarlyStopping


class TargetClientES(BaseClient):
    """
    Target client with improved fine-tuning strategies.

    Supported ft_mode:
        full:
            fixed-epoch full fine-tuning

        early_stopping:
            full fine-tuning + validation early stopping

        regularized:
            full fine-tuning + regularization + validation early stopping

        predictor_only:
            only update stPredictor + validation early stopping

        private_only:
            update spatialModel + stPare + stPredictor + validation early stopping

        staged_reg_es:
            Stage 1: predictor-only warmup
            Stage 2: private modules + predictor, with regularization and validation early stopping

    This file is designed to work with:
        utils/data_utils_es.py
        utils/early_stopping.py
        trainers/pFedCTP_ES.py
        run_pFedCTP_ES.py
    """

    def __init__(
        self,
        id,
        name,
        train_dataset,
        val_dataset,
        test_dataset,
        options,
        model,
    ):
        super(TargetClientES, self).__init__(
            id=id,
            name=name,
            dataset=train_dataset,
            options=options,
            model=model,
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.options["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.options["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.options["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    # ============================================================
    # Basic helpers
    # ============================================================

    def _prepare_data(self, data, dataset):
        data.str_init = dataset.str_init
        data.edge_index = dataset.edge_index_list
        data = data.to(self.device)
        return data

    def _reset_requires_grad(self):
        for p in self.model.parameters():
            p.requires_grad = True

    def _freeze_all(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def _unfreeze_module(self, module_name, strict=False):
        module = getattr(self.model, module_name, None)

        if module is None:
            msg = f"Module '{module_name}' not found."

            if strict:
                raise AttributeError(msg)

            print(f"  [Warning] {msg} Skipped.")
            return False

        for p in module.parameters():
            p.requires_grad = True

        return True

    def _set_trainable_modules(self, module_names, strict=False):
        self._freeze_all()

        found_any = False

        for module_name in module_names:
            found = self._unfreeze_module(module_name, strict=strict)
            found_any = found_any or found

        if not found_any:
            self._reset_requires_grad()
            raise RuntimeError(
                f"No trainable modules found from: {module_names}. "
                "Please check model module names."
            )

    def _make_anchor_state(self):
        anchor_state = {}

        for name, param in self.model.named_parameters():
            anchor_state[name] = param.detach().clone().to(self.device)

        return anchor_state

    def _regularization_loss(self, anchor_state):
        """
        Parameter distance to the model before target fine-tuning.

        Default design:
            reg_exclude_predictor = 1
                Do not regularize stPredictor.
                This lets the final prediction head adapt more freely.

            reg_normalize = 1
                Use average squared distance.
                This makes reg_lambda easier to tune.
        """

        if anchor_state is None:
            return torch.tensor(0.0, device=self.device)

        exclude_predictor = int(self.options.get("reg_exclude_predictor", 1))
        normalize = int(self.options.get("reg_normalize", 1))

        reg_loss = torch.tensor(0.0, device=self.device)
        total_numel = 0

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if exclude_predictor == 1 and "stPredictor" in name:
                continue

            if name not in anchor_state:
                continue

            anchor_param = anchor_state[name]
            reg_loss = reg_loss + torch.sum((param - anchor_param) ** 2)
            total_numel += param.numel()

        if normalize == 1 and total_numel > 0:
            reg_loss = reg_loss / total_numel

        return reg_loss

    def _make_optimizer(self):
        """
        Create Adam optimizer with optional layer-wise learning rates.

        predictor_lr_mult:
            learning rate multiplier for stPredictor

        private_lr_mult:
            learning rate multiplier for spatialModel and stPare

        other_lr_mult:
            learning rate multiplier for other trainable parameters
        """

        base_lr = float(self.inner_lr)

        predictor_lr_mult = float(self.options.get("predictor_lr_mult", 1.0))
        private_lr_mult = float(self.options.get("private_lr_mult", 1.0))
        other_lr_mult = float(self.options.get("other_lr_mult", 1.0))

        predictor_params = []
        private_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if "stPredictor" in name:
                predictor_params.append(param)
            elif "spatialModel" in name or "stPare" in name:
                private_params.append(param)
            else:
                other_params.append(param)

        param_groups = []

        if len(predictor_params) > 0:
            param_groups.append(
                {
                    "params": predictor_params,
                    "lr": base_lr * predictor_lr_mult,
                }
            )

        if len(private_params) > 0:
            param_groups.append(
                {
                    "params": private_params,
                    "lr": base_lr * private_lr_mult,
                }
            )

        if len(other_params) > 0:
            param_groups.append(
                {
                    "params": other_params,
                    "lr": base_lr * other_lr_mult,
                }
            )

        if len(param_groups) == 0:
            raise RuntimeError("No trainable parameters found for optimizer.")

        return optim.Adam(param_groups)

    # ============================================================
    # Training / validation / testing
    # ============================================================

    def _train_one_epoch(
        self,
        optimizer,
        use_regularization=False,
        anchor_state=None,
        reg_lambda=0.0,
    ):
        self.model.train()

        total_loss_sum = 0.0
        pred_loss_sum = 0.0
        reg_loss_sum = 0.0
        count = 0

        for data in self.train_dataloader:
            data = self._prepare_data(data, self.train_dataset)

            optimizer.zero_grad()

            out = self.model(data)
            pred_loss = self.calculate_loss(out, data.y)

            if use_regularization:
                reg_loss = self._regularization_loss(anchor_state)
                total_loss = pred_loss + reg_lambda * reg_loss
            else:
                reg_loss = torch.tensor(0.0, device=self.device)
                total_loss = pred_loss

            total_loss.backward()
            optimizer.step()

            total_loss_sum += total_loss.item()
            pred_loss_sum += pred_loss.item()
            reg_loss_sum += reg_loss.item()
            count += 1

        avg_total_loss = total_loss_sum / max(count, 1)
        avg_pred_loss = pred_loss_sum / max(count, 1)
        avg_reg_loss = reg_loss_sum / max(count, 1)

        return avg_total_loss, avg_pred_loss, avg_reg_loss

    @torch.no_grad()
    def _evaluate_loss(self, dataloader, dataset):
        """
        Validation and test loss only use prediction loss.

        We do not include regularization loss here, because validation should
        reflect forecasting performance.
        """

        self.model.eval()

        total_loss = 0.0
        count = 0

        for data in dataloader:
            data = self._prepare_data(data, dataset)

            out = self.model(data)
            loss = self.calculate_loss(out, data.y)

            total_loss += loss.item()
            count += 1

        return total_loss / max(count, 1)

    @torch.no_grad()
    def target_test(self):
        self.model.eval()

        outputs = None
        y_label = None

        for data in self.test_dataloader:
            data = self._prepare_data(data, self.test_dataset)

            out = self.model(data)

            if outputs is None:
                outputs = out
                y_label = data.y
            else:
                outputs = torch.cat((outputs, out))
                y_label = torch.cat((y_label, data.y))

        q_metric = metric_func(outputs, y_label)
        return q_metric

    # ============================================================
    # Fine-tuning entrance
    # ============================================================

    def target_dataloader(self):
        ft_mode = self.options.get("ft_mode", "early_stopping")

        print("\n[Target Fine-tuning]")
        print(f"  city: {self.name}")
        print(f"  ft_mode: {ft_mode}")

        if ft_mode == "full":
            return self._ft_full()

        if ft_mode == "early_stopping":
            self._reset_requires_grad()
            return self._ft_with_validation(
                use_regularization=False,
                description="full fine-tuning + validation early stopping",
            )

        if ft_mode == "regularized":
            self._reset_requires_grad()
            return self._ft_with_validation(
                use_regularization=True,
                description="regularized fine-tuning + validation early stopping",
            )

        if ft_mode == "predictor_only":
            return self._ft_predictor_only()

        if ft_mode == "private_only":
            return self._ft_private_only()

        if ft_mode == "staged_reg_es":
            return self._ft_staged_reg_es()

        raise ValueError(
            f"Unknown ft_mode: {ft_mode}. "
            "Please use one of: full, early_stopping, regularized, "
            "predictor_only, private_only, staged_reg_es."
        )

    # ============================================================
    # Method 1: Full fixed-epoch fine-tuning
    # ============================================================

    def _ft_full(self):
        self._reset_requires_grad()

        max_epochs = int(self.options["target_epochs"])

        print("  strategy: full fixed-epoch fine-tuning")
        print(f"  target_epochs: {max_epochs}")

        optimizer = self._make_optimizer()

        for epoch in range(1, max_epochs + 1):
            total_loss, pred_loss, reg_loss = self._train_one_epoch(
                optimizer=optimizer,
                use_regularization=False,
            )

            if epoch == 1 or epoch % 5 == 0:
                print(
                    f"  [Epoch {epoch:03d}] "
                    f"train_loss={pred_loss:.6f}"
                )

        test_loss = self._evaluate_loss(
            self.test_dataloader,
            self.test_dataset,
        )

        print(f"  Final test_loss: {test_loss:.6f}")

        q_metric = self.target_test()
        return q_metric

    # ============================================================
    # Method 2: Predictor-only + validation early stopping
    # ============================================================

    def _ft_predictor_only(self):
        print("  strategy: predictor-only + validation early stopping")
        print("  trainable module: stPredictor")

        self._set_trainable_modules(
            module_names=["stPredictor"],
            strict=True,
        )

        try:
            result = self._ft_with_validation(
                use_regularization=False,
                description="predictor-only + validation early stopping",
            )
        finally:
            self._reset_requires_grad()

        return result

    # ============================================================
    # Method 3: Private-only + validation early stopping
    # ============================================================

    def _ft_private_only(self):
        print("  strategy: private-only + validation early stopping")
        print("  trainable modules: spatialModel, stPare, stPredictor")

        self._set_trainable_modules(
            module_names=["spatialModel", "stPare", "stPredictor"],
            strict=False,
        )

        try:
            result = self._ft_with_validation(
                use_regularization=False,
                description="private-only + validation early stopping",
            )
        finally:
            self._reset_requires_grad()

        return result

    # ============================================================
    # Method 4 / 5: Validation training core
    # ============================================================

    def _ft_with_validation(
        self,
        use_regularization=False,
        description="validation fine-tuning",
    ):
        max_epochs = int(self.options.get("target_epochs", 50))

        patience = self.options.get(
            "early_stop_patience",
            self.options.get("es_patience", 10),
        )

        if patience is None:
            patience = self.options.get("es_patience", 10)

        patience = int(patience)

        min_delta = float(self.options.get("early_stop_min_delta", 0.0))
        print_every = int(self.options.get("early_stop_print_every", 1))

        reg_lambda = float(self.options.get("reg_lambda", 1.0))

        if use_regularization:
            anchor_state = self._make_anchor_state()
        else:
            anchor_state = None
            reg_lambda = 0.0

        print(f"  strategy: {description}")
        print(f"  max_epochs: {max_epochs}")
        print(f"  patience: {patience}")
        print(f"  min_delta: {min_delta}")
        print(f"  train samples: {len(self.train_dataset)}")
        print(f"  val samples:   {len(self.val_dataset)}")
        print(f"  test samples:  {len(self.test_dataset)}")

        if use_regularization:
            print(f"  reg_lambda: {reg_lambda}")
            print(f"  reg_exclude_predictor: {self.options.get('reg_exclude_predictor', 1)}")
            print(f"  reg_normalize: {self.options.get('reg_normalize', 1)}")

        optimizer = self._make_optimizer()

        early_stopper = EarlyStopping(
            patience=patience,
            min_delta=min_delta,
        )

        for epoch in range(1, max_epochs + 1):
            total_loss, pred_loss, reg_loss = self._train_one_epoch(
                optimizer=optimizer,
                use_regularization=use_regularization,
                anchor_state=anchor_state,
                reg_lambda=reg_lambda,
            )

            val_loss = self._evaluate_loss(
                self.val_dataloader,
                self.val_dataset,
            )

            if epoch == 1 or epoch % print_every == 0:
                if use_regularization:
                    print(
                        f"  [Target Epoch {epoch:03d}] "
                        f"total_loss={total_loss:.6f}, "
                        f"pred_loss={pred_loss:.6f}, "
                        f"reg_loss={reg_loss:.6f}, "
                        f"lambda_reg={reg_lambda * reg_loss:.6f}, "
                        f"val_loss={val_loss:.6f}"
                    )
                else:
                    print(
                        f"  [Target Epoch {epoch:03d}] "
                        f"train_loss={pred_loss:.6f}, "
                        f"val_loss={val_loss:.6f}"
                    )

            should_stop = early_stopper.step(
                val_loss,
                self.model,
                epoch,
            )

            if should_stop:
                print(
                    f"  Early stopping triggered at epoch {epoch}. "
                    f"Best epoch = {early_stopper.best_epoch}, "
                    f"best val_loss = {early_stopper.best_score:.6f}"
                )
                break

        self.model = early_stopper.load_best_model(self.model)

        test_loss = self._evaluate_loss(
            self.test_dataloader,
            self.test_dataset,
        )

        print(f"  Final test_loss after loading best model: {test_loss:.6f}")

        q_metric = self.target_test()
        return q_metric

    # ============================================================
    # Improved method: staged regularized early stopping
    # ============================================================

    def _ft_staged_reg_es(self):
        """
        Proposed improved strategy.

        Stage 1:
            Train only stPredictor for a few epochs.
            This lets the prediction head adapt to target city quickly.

        Stage 2:
            Train spatialModel + stPare + stPredictor.
            Use regularization and validation early stopping.
            This avoids over-updating the representation modules.
        """

        warmup_epochs = int(self.options.get("warmup_epochs", 5))

        print("  strategy: staged regularized early stopping")
        print(f"  warmup_epochs: {warmup_epochs}")

        # ------------------------------------------------------------
        # Stage 1: predictor-only warmup
        # ------------------------------------------------------------

        if warmup_epochs > 0:
            print("  [Stage 1] predictor-only warmup")

            self._set_trainable_modules(
                module_names=["stPredictor"],
                strict=True,
            )

            warmup_optimizer = self._make_optimizer()

            for epoch in range(1, warmup_epochs + 1):
                total_loss, pred_loss, reg_loss = self._train_one_epoch(
                    optimizer=warmup_optimizer,
                    use_regularization=False,
                )

                print(
                    f"  [Warmup Epoch {epoch:03d}] "
                    f"train_loss={pred_loss:.6f}"
                )

            warmup_val_loss = self._evaluate_loss(
                self.val_dataloader,
                self.val_dataset,
            )

            print(f"  [After Warmup] val_loss={warmup_val_loss:.6f}")

        # ------------------------------------------------------------
        # Stage 2: private modules + predictor with regularization
        # ------------------------------------------------------------

        print("  [Stage 2] private modules + predictor + regularization + validation")

        self._set_trainable_modules(
            module_names=["spatialModel", "stPare", "stPredictor"],
            strict=False,
        )

        try:
            result = self._ft_with_validation(
                use_regularization=True,
                description=(
                    "stage 2: private modules + predictor "
                    "+ regularization + validation early stopping"
                ),
            )
        finally:
            self._reset_requires_grad()

        return result