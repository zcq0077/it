class EarlyStopping:
    def __init__(self, patience=3, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, monitor_score, model):
        if self.best_score is None:
            self.best_score = monitor_score
            self.save_checkpoint(monitor_score, model)
        elif monitor_score > self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(monitor_score, model)
            self.best_score = monitor_score
            self.counter = 0

    def save_checkpoint(self, monitor_score, model):
        if self.verbose:
            if self.best_score is None:
                print(f'Monitor score initialized ({monitor_score:.6f}).')
            else:
                print(f'Monitor score improved ({self.best_score:.6f} --> {monitor_score:.6f}).')
        # torch.save(model.state_dict(), 'checkpoint.pt')
