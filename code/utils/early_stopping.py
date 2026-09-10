import copy


class EarlyStopping:
    """
    Validation-based early stopping.

    監控 validation loss：
    - 如果 validation loss 有變好，就保存目前模型參數
    - 如果連續 patience 個 epoch 沒變好，就停止 fine-tuning
    """

    def __init__(self, patience=8, min_delta=0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)

        self.best_score = None
        self.best_epoch = 0
        self.best_state_dict = None

        self.counter = 0
        self.early_stop = False

    def step(self, current_score, model, epoch):
        """
        Args:
            current_score: validation loss
            model: current model
            epoch: current epoch number

        Returns:
            True means stop training.
        """

        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.counter = 0
            return False

        improved = current_score < self.best_score - self.min_delta

        if improved:
            self.best_score = current_score
            self.best_epoch = epoch
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def load_best_model(self, model):
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
        return model