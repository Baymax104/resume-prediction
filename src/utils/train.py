# -*- coding: UTF-8 -*-
import time
from pathlib import Path

import numpy as np
import torch
from prettytable import PrettyTable


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def save_model(model, model_name, log_dir):
    model_dir = Path(log_dir) / 'model'
    if not model_dir.is_dir():
        model_dir.mkdir(exist_ok=True)
    model_name = f'{model_name}-{time.strftime("%Y%m%d-%H%M%S")}.pt'
    model_path = model_dir / model_name
    torch.save(model.state_dict(), model_path)


def load_model(model, model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise ValueError(f'Model {str(model_path)} does not exist.')
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)


def log_train_settings(settings):
    table = PrettyTable(header=False)
    table.add_row(["Epoch", settings.train.epochs])
    table.add_row(["Batch Size", settings.data.batch_size])
    table.add_row(["Learning Rate", settings.train.lr])
    table.add_row(["Weight Decay", settings.train.weight_decay])
    print(table)
