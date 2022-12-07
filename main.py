import os

import torch
import torch.nn as nn
import torchinfo

from utils import DirectorySetup, create_dataloaders
from models import RockPaperScissor_Model_V1
from trainer import Trainer
import config


dir_setup = DirectorySetup(config.RAW_DATA_DIR)
dir_setup.setup(config.DATA_DIR)
print(dir_setup)

train_dir = os.path.join(config.DATA_DIR, "train")
valid_dir = os.path.join(config.DATA_DIR, "valid")
test_dir  = os.path.join(config.DATA_DIR, "test" )

train_dl, valid_dl, test_dl = create_dataloaders(train_dir, valid_dir, test_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_v1 = RockPaperScissor_Model_V1()
torchinfo.summary(model_v1, input_size=[1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE])

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_v1.parameters(), lr=1e-3)

trainer_obj = Trainer(model_v1, criterion, optimizer, device)

model_v1_results = trainer_obj.train(train_dl, valid_dl, epochs=config.EPOCHS)

trainer_obj.plot_model_results()

trainer_obj.save("model_v1.pt")

loaded_model_v1 = trainer_obj.load("model_v1.pt")
torchinfo.summary(loaded_model_v1, input_size=[1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE])

