from src.utils import load_data, load_pre_embedded
from sklearn.model_selection import GroupShuffleSplit
import torch
import numpy as np
import torchmetrics
from tqdm import tqdm
import pickle
import os


class Trainer:
    def __init__(self, config):
        self.config = config

        # Loads messages and message embeddings
        self.messages, self.labels, self.game_ids = load_data(self.config.data.pickle)

        self.message_encodings, self.labels, self.game_ids = load_pre_embedded(self.config.data.folder)

        # Splits data as specified in the config
        match self.config.data.split_by:
            case 'userID':
                self.split_by_user_id()
            case 'gameID':
                self.split_by_game_id()
            case _:
                raise NotImplementedError(f"Split by {self.config.data.split_by} is not implemented")
        
        # Creates train and validation dataset
        self.X_train, self.y_train = torch.tensor(np.array([self.message_encodings[i] for i in self.train_idx]), dtype=torch.float32), torch.tensor(np.array([self.labels[i] for i in self.train_idx]), dtype=torch.float32)
        self.X_val, self.y_val = torch.tensor(np.array([self.message_encodings[i] for i in self.val_idx]), dtype=torch.float32), torch.tensor(np.array([self.labels[i] for i in self.val_idx]), dtype=torch.float32)

        self.train_set = torch.utils.data.TensorDataset(self.X_train, self.y_train)
        self.val_set = torch.utils.data.TensorDataset(self.X_val, self.y_val)

        # Creates train and validation dataloader
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.config.training.batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_set, batch_size=self.config.validation.batch_size, shuffle=False)

        self.device = self.config.device


    def train(self, model):
        """
        Trains the model using the dataset and specifications given during initialization of the trainer

        Args:
            model: The model to be trained.

        Returns:
            model: The trained model.
            train_accuracy: Training accuracy of the model per epoch (List)
            validation_accuracy: Validation accuracy of the model per epoch (List)
            train_loss: Training loss values of the model per epoch (List)
            validation_loss: Validation loss values of the model per epoch (List)
        """
        train_config = self.config.training
        if 'scheduler' in self.config:
            pass
        else:
            scheduler = None

        match self.config.optimizer.type:
            case 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=self.config.optimizer.lr, weight_decay=self.config.optimizer.weight_decay)
            case 'adamW':
                optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.optimizer.lr, weight_decay=self.config.optimizer.weight_decay)
            case _:
                raise NotImplementedError(f"Optimizer {self.config.optimizer} is not implemented")
            
        match train_config.loss_fn:
            case 'cross_entropy':
                loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=train_config.label_smoothing)
            case _:
                raise NotImplementedError(f"Loss function {train_config.loss_fn} is not implemented")

        model.train()
        model.to(self.device)
        train_acc = []
        train_losses = []
        validation_acc = []
        validation_losses = []
        
        model.eval()
        task="binary"
        accuracy = torchmetrics.Accuracy(num_classes=2, task=task)
        train_loss = 0
        with torch.no_grad():
            for features, labels in self.train_loader:
                output = model(features.to(self.device))
                accuracy(torch.argmax(output.to("cpu"), dim=1), torch.argmax(labels, dim=1).to("cpu"))
                train_loss += loss_fn(output.to("cpu"), labels.to("cpu")).item()
        model.train()

        train_acc.append(accuracy.compute())
        train_losses.append(train_loss/len(self.train_loader))
        val_accuracy, val_loss = self.evaluate(model, loss_fn)
        validation_acc.append(val_accuracy)
        validation_losses.append(val_loss)
        
        for epoch in tqdm(range(train_config.epochs), desc=self.config.name):
            task="binary"
            accuracy = torchmetrics.Accuracy(num_classes=2, task=task)
            train_loss = 0
            for features, labels in self.train_loader:
                optimizer.zero_grad()
                output = model(features.to(self.device))
                loss = loss_fn(output, labels.to(self.device))
                loss.backward()
                optimizer.step()
                accuracy(torch.argmax(output.to("cpu"), dim=1), torch.argmax(labels.to("cpu"), dim=1))
                train_loss += loss.item()
                
            val_accuracy, val_loss = self.evaluate(model, loss_fn)

            train_acc.append(accuracy.compute())
            train_losses.append(train_loss/len(self.train_loader))
            validation_acc.append(val_accuracy)
            validation_losses.append(val_loss)
        model.eval()
        model.to("cpu")
        return model, train_acc, validation_acc, train_losses, validation_losses
    
    def evaluate(self, model, loss_fn):
        """
        Evaluates the model using loss_fn and the specifications given during initialization of the trainer

        Args:
            model (int): Model to be evaluated
            loss_fn: Torch loss function to be used

        Returns:
            accuracy: Validation accuracy of the model
            validation_loss: Average validation loss
        """
        model.eval()
        task="binary"
        accuracy = torchmetrics.Accuracy(num_classes=2, task=task)
        validation_loss = 0
        with torch.no_grad():
            for features, labels in self.val_loader:
                output = model(features.to(self.device))
                accuracy(torch.argmax(output.to("cpu"), dim=1), torch.argmax(labels, dim=1).to("cpu"))
                validation_loss += loss_fn(output.to("cpu"), labels.to("cpu")).item()
        model.train()
        return accuracy.compute(), validation_loss/len(self.val_loader)
    
    def split_by_user_id(self):
        """
        Splits the data using unique user_ids for train and eval set.
        """
        # Load user ids
        with open(os.path.join("data", "user_ids.pkl"), "rb") as f:
            self.user_ids = pickle.load(f)

        numpy_user_ids = np.array(self.user_ids, dtype=str)

        # Extracts ai message indices and human message indices
        ai_mask = numpy_user_ids == "0"
        human_mask = numpy_user_ids != "0"

        ai_indices = np.where(ai_mask)[0]
        human_indices = np.where(human_mask)[0]

        human_user_ids = numpy_user_ids[human_indices]

        # Splits the human written messages by user id
        gss = GroupShuffleSplit(n_splits=1, test_size=self.config.data.test_size, random_state=42)
        human_train_subidx, human_val_subidx = next(gss.split(human_indices, groups=human_user_ids))

        human_train_idx = human_indices[human_train_subidx]
        human_val_idx = human_indices[human_val_subidx]
        
        self.human_train_idx = human_train_idx
        self.human_val_idx = human_val_idx

        # Calculates the ratio of ai written messages in the original data
        total_len = len(numpy_user_ids)
        ai_ratio = len(ai_indices) / total_len

        # Calculates number of ai messages for train and eval set
        expected_train_len = len(human_train_idx) / (1 - ai_ratio)
        expected_val_len = len(human_val_idx) / (1 - ai_ratio)

        # Randomly selects the calculated number of ai messages for train and eval
        n_ai_train = int(round(expected_train_len * ai_ratio))
        n_ai_val = int(round(expected_val_len * ai_ratio))

        n_ai_train = min(n_ai_train, len(ai_indices))
        n_ai_val = min(n_ai_val, len(ai_indices) - n_ai_train)

        ai_indices_shuffled = np.random.RandomState(seed=42).permutation(ai_indices)
        ai_train_idx = ai_indices_shuffled[:n_ai_train]
        ai_val_idx = ai_indices_shuffled[n_ai_train:n_ai_train + n_ai_val]
        
        # Merges the ai messages with the human messages
        self.train_idx = np.sort(np.concatenate([human_train_idx, ai_train_idx]))
        self.val_idx = np.sort(np.concatenate([human_val_idx, ai_val_idx]))

    def split_by_game_id(self):
        """
        Splits the data using unique game ids for train and eval set.
        """
        self.gss = GroupShuffleSplit(n_splits=1, test_size=self.config.data.test_size, random_state=42)
        self.train_idx, self.val_idx = next(self.gss.split(self.messages, self.labels, self.game_ids))
