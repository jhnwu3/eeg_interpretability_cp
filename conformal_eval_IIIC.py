import torch
import torch
import pytorch_lightning as pl
from models import *
from interpret.chefer import *
from models.st_transformer import *
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from models.pytorch_lightning import *
from data import *
from uq.conformal import *
from pyhealth.metrics import multiclass_metrics_fn
# my training hyperparameters

batch_size = 512
num_workers = 32
lr = 0.01
total_epochs = 50

# my model hyperparameters
emb_size = 256
depth = 6 
dropout = 0.5
num_heads = 8
patch_kernel_length = 11  # cqi = 15 - UNUSED
stride = 11  # cqi = 8 - UNUSED

train_loader, test_loader, val_loader, cal_loader = prepare_IIIC_cal_dataloader(batch_size=batch_size, num_workers=num_workers, drop_last=True)
signal, label = train_loader.dataset[0]
# print(signal)
# exit(0)
# define the model for training - STT transformer
st_transformer = STTransformer(emb_size=emb_size, 
                                depth=depth,
                                n_classes=6, 
                                channel_length=2000,
                                dropout=dropout, 
                                num_heads=num_heads,
                                kernel_size=11, 
                                stride=11,
                                kernel_size2=11,
                                stride2=11)

st_transformer.load_state_dict(torch.load("saved_weights/IIIC_st_transformer_conformal_c11s11c5s5.pt"))
st_transformer = st_transformer.cuda()

# evaluate on test_loader
y_true = []
y_prob = []

for signal, label in test_loader:
    prob = st_transformer(signal.cuda()).softmax(1)
    # append every "batch" 
    for i in range(prob.size()[0]):
        y_true.append(label[i])
        y_prob.append(prob[i].detach().cpu().numpy())

y_true = np.array(y_true)
y_prob = np.array(y_prob)

print(y_true.shape)
print(y_prob.shape)
print(multiclass_metrics_fn(y_true, y_prob, metrics=["accuracy", "roc_auc_macro_ovr"]))

# calibration step after validation and what not.

# now do with the conformity scores



# train model with pytlightning
