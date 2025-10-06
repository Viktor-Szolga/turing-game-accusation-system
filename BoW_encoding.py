
import torch
import numpy as np
import pickle
from openai import OpenAI
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

with open("data/190325_data.pkl", "rb") as f:
    chat_data = pickle.load(f)
    
# Remove service chat
del chat_data[-1000]

messages = []
labels = []
game_ids = []
user_ids = []
languages = []
for game_id, game_data in chat_data.items():
    for message in game_data["messages"]:
        if message["userID"] == "GameMaster":
            if "won" in message["message"] or "surrendered" in message["message"] or "canceled" in message["message"] or "lost" in message["message"] or "timed out" in message["message"] or "disconnected" in message["message"]:
                break
            else:
                continue
        messages.append(message["message"])
        labels.append([int(not message["botID"]), message["botID"]])
        game_ids.append(message["gameID"])
        user_ids.append(message["userID"])
        if game_data["language"] is None:
            languages.append("en")
        else:
            languages.append(game_data["language"])


all_words = [word.lower() for message in messages for word in message.split(" ")]
print(len(all_words))
most_common_1024 = [s for s, _ in Counter(all_words).most_common(1024)]
message_encodings = np.zeros((len(messages), 1024))
for i, message in enumerate(messages):
    for word in message.split(" "):
        if word.lower() in most_common_1024:
            index = most_common_1024.index(word.lower())
            message_encodings[i][index] += 1

with open(os.path.join("data", "BoW_most_common.pkl"), "wb") as f:
    pickle.dump(most_common_1024, f)
with open(os.path.join("data", "BoW_encodings.pkl"), "wb") as f:
    pickle.dump(message_encodings, f)
    
print(sum(sum(message_encodings)))