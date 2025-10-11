from src.utils import load_data, load_pre_embedded
from sklearn.model_selection import GroupShuffleSplit
import torch
import numpy as np
import torchmetrics
from tqdm import tqdm
import pickle
import os


class Trainer:
    def __init__(self, config, run_name):
        self.config = config
        self.device = self.config.device
        self.run_name = run_name

        # Load data
        self.messages, self.labels, self.game_ids = load_data(self.config.data.pickle)
        self.message_encodings, self.labels, self.game_ids = load_pre_embedded(self.config.data.folder)
        self.labels = np.array(self.labels)

        # Fix labels [0,2] -> [0,1]
        for i, label in enumerate(self.labels):
            if self.labels[i][1] == 2:
                self.labels[i] = [0, 1]

        # Split dataset
        match self.config.data.split_by:
            case 'userID':
                self.split_by_user_id()
            case 'gameID':
                self.split_by_game_id()
            case _:
                raise NotImplementedError(f"Split by {self.config.data.split_by} not implemented")

        # Move data to GPU immediately (fastest if it fits)
        self.X_train = torch.tensor(np.array([self.message_encodings[i] for i in self.train_idx]), dtype=torch.float32, device=self.device)
        self.y_train = torch.tensor(np.array([self.labels[i] for i in self.train_idx]), dtype=torch.float32, device=self.device)
        self.X_val = torch.tensor(np.array([self.message_encodings[i] for i in self.val_idx]), dtype=torch.float32, device=self.device)
        self.y_val = torch.tensor(np.array([self.labels[i] for i in self.val_idx]), dtype=torch.float32, device=self.device)

        # TensorDatasets & DataLoaders
        self.train_set = torch.utils.data.TensorDataset(self.X_train, self.y_train)
        self.val_set = torch.utils.data.TensorDataset(self.X_val, self.y_val)
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.config.training.batch_size, shuffle=True, num_workers=0)
        self.val_loader = torch.utils.data.DataLoader(self.val_set, batch_size=self.config.validation.batch_size, shuffle=False, num_workers=0)

    def train(self, model):
        train_config = self.config.training

        # Optimizer
        match self.config.optimizer.type:
            case 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=self.config.optimizer.lr, weight_decay=self.config.optimizer.weight_decay)
            case 'adamW':
                optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.optimizer.lr, weight_decay=self.config.optimizer.weight_decay)
            case _:
                raise NotImplementedError(f"Optimizer {self.config.optimizer.type} not implemented")

        # Weighted CrossEntropy
        label_indices = np.argmax(self.labels, axis=1)
        class_counts = np.bincount(label_indices)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=train_config.label_smoothing, weight=class_weights)

        model.to(self.device)
        model.train()

        train_acc_list, train_loss_list = [], []
        val_acc_list, val_loss_list = [], []

        # Early stopping parameters
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        patience_limit = 10  # stop if no improvement for 10 epochs
        stopped = False

        # Make sure output directory exists
        os.makedirs("checkpoints", exist_ok=True)
        model_path = os.path.join("checkpoints", f"{self.run_name}.pth")
        info_path = os.path.join("checkpoints", "best_model_info.txt")
        
        with torch.no_grad():
            # Training set metrics
            train_loss = 0
            accuracy_train = torchmetrics.Accuracy(num_classes=2, task="binary").to(self.device)
            for features, labels in self.train_loader:
                outputs = model(features)
                train_loss += loss_fn(outputs, labels).item()
                accuracy_train(torch.argmax(outputs, dim=1), torch.argmax(labels, dim=1))
            train_loss_list.append(train_loss / len(self.train_loader))
            train_acc_list.append(accuracy_train.compute().item())

            # Validation set metrics
            val_acc, val_loss = self.evaluate(model, loss_fn)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc.item())

        for epoch in tqdm(range(train_config.epochs), desc=self.config.name):
            train_loss = 0
            accuracy = torchmetrics.Accuracy(num_classes=2, task="binary").to(self.device)

            for features, labels in self.train_loader:
                optimizer.zero_grad()
                outputs = model(features)  # Already on GPU
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                accuracy(torch.argmax(outputs, dim=1), torch.argmax(labels, dim=1))

            train_acc_list.append(accuracy.compute().item())
            train_loss_list.append(train_loss / len(self.train_loader))

            val_acc, val_loss = self.evaluate(model, loss_fn)
            val_acc_list.append(val_acc.item())
            val_loss_list.append(val_loss)

            # Early stopping logic
            if val_loss < best_val_loss and not stopped:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), model_path)

                # Write info file
            else:
                patience_counter += 1

            if patience_counter >= patience_limit and not stopped:
                with open(info_path, "a") as f:
                    f.write(f"{self.run_name} | {best_val_loss:.6f} | {best_epoch}\n")
                stopped = True

        model.eval()
        model.to("cpu")
        return model, train_acc_list, val_acc_list, train_loss_list, val_loss_list

    def evaluate(self, model, loss_fn):
        model.eval()
        accuracy = torchmetrics.Accuracy(num_classes=2, task="binary").to(self.device)
        val_loss = 0

        with torch.no_grad():
            for features, labels in self.val_loader:
                outputs = model(features)  # Already on GPU
                val_loss += loss_fn(outputs, labels).item()
                accuracy(torch.argmax(outputs, dim=1), torch.argmax(labels, dim=1))

        model.train()
        return accuracy.compute(), val_loss / len(self.val_loader)

    def split_by_user_id(self):
        with open(os.path.join("data", "user_ids.pkl"), "rb") as f:
            self.user_ids = pickle.load(f)

        numpy_user_ids = np.array(self.user_ids, dtype=str)
        ai_mask = numpy_user_ids == "0"
        human_mask = numpy_user_ids != "0"

        ai_indices = np.where(ai_mask)[0]
        human_indices = np.where(human_mask)[0]
        human_user_ids = numpy_user_ids[human_indices]

        gss = GroupShuffleSplit(n_splits=1, test_size=self.config.data.test_size, random_state=42)
        human_train_subidx, human_val_subidx = next(gss.split(human_indices, groups=human_user_ids))

        human_train_idx = human_indices[human_train_subidx]
        human_val_idx = human_indices[human_val_subidx]

        total_len = len(numpy_user_ids)
        ai_ratio = len(ai_indices) / total_len

        expected_train_len = len(human_train_idx) / (1 - ai_ratio)
        expected_val_len = len(human_val_idx) / (1 - ai_ratio)

        n_ai_train = min(int(round(expected_train_len * ai_ratio)), len(ai_indices))
        n_ai_val = min(int(round(expected_val_len * ai_ratio)), len(ai_indices) - n_ai_train)

        ai_indices_shuffled = np.random.RandomState(seed=42).permutation(ai_indices)
        ai_train_idx = ai_indices_shuffled[:n_ai_train]
        ai_val_idx = ai_indices_shuffled[n_ai_train:n_ai_train + n_ai_val]

        self.train_idx = np.sort(np.concatenate([human_train_idx, ai_train_idx]))
        self.val_idx = np.sort(np.concatenate([human_val_idx, ai_val_idx]))

    def split_by_game_id(self):
        gss = GroupShuffleSplit(n_splits=1, test_size=self.config.data.test_size, random_state=42)
        self.train_idx, self.val_idx = next(gss.split(self.messages, self.labels, self.game_ids))
