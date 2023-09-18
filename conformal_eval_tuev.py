import torch
import torch
import csv 
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


def write_to_csv(path, list1, list2, column_names):
     with open(path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(column_names)
        for values in zip(list1, list2):
            writer.writerow(values)
     csv_file.close()

class_labels = {
    0 : 'SPSW', 1 : 'GPED', 2 : 'PLED', 3 : 'EYEM', 4: 'ARTF', 5 :'BCKG'
}

# my model hyperparameters
emb_size = 256
depth = 4 
dropout = 0.5
num_heads = 8
model = STTransformer(emb_size=emb_size, 
                                   depth=depth,
                                   n_classes=6, 
                                   channel_length=1000,
                                   dropout=dropout, 
                                   num_heads=num_heads,
                                   kernel_size=11, 
                                   stride=11,
                                   kernel_size2=5,
                                   stride2=5)

model.load_state_dict(torch.load("saved_weights/st_transformer_conformal_c11s11c5s5.pt"))


# my training hyperparameters
sampling_rate = 200
batch_size = 1 # code can only interpret 1 at at time
num_workers = 32
train_loader, test_loader, val_loader, cal_loader = prepare_TUEV_cal_dataloader(sampling_rate=sampling_rate, batch_size=batch_size, num_workers=num_workers)

# get interpreter
interpreter = STTransformerInterpreter(model=model)


alphas = [0.1, 0.2, 0.3, 0.4, 0.5] 
column_names = ["alpha", "q_hat"]
signal, label = test_loader.dataset[0]
signal = signal.unsqueeze(0)

# q_hats = []
# print("E SCORE")
# for alpha in alphas:
#     q_hat = get_calibration_cutoff(model=model, 
#                         calibration_dataloader=cal_loader,
#                             interpreter=interpreter, 
#                             cal_scoring_function=cal_e_score, alpha=alpha)
#     q_hats.append(q_hat)

#     # q_hat = 0.9984634 # alpha=0.0001
#     # q_hat = -7.200241e-05 # alpha = 0.5 
#     # 0.00020742416 for q_hat for alpha = 0.3
#     print("alpha, q_hat, prediction set, prediction set length")
    
    
#     pred_set = generate_prediction_set(signal, model, q_hat, interpreter, e_score, class_index=label)
#     print(alpha, ",",q_hat, ",", pred_set, ",", len(pred_set))

# write_to_csv("uq/q_hat/e_score.csv", alphas, q_hats, column_names=column_names)
#     # pred_set = print_prediction_set(pred_set, label_map=class_labels)




# q_hats = []
# # we should run experiments across e_scores, softmax, and road scores, and maybe get an average across the entire test dataset
# print("SOFTMAX")
# for alpha in alphas:
#     q_hat = get_calibration_cutoff(model=model, 
#                         calibration_dataloader=cal_loader,
#                             interpreter=interpreter, 
#                             cal_scoring_function=cal_softmax, alpha=alpha)
#     q_hats.append(q_hat)
#     print("alpha, q_hat, prediction set, prediction set length")
#     pred_set = generate_prediction_set(signal, model, q_hat, interpreter, softmax, class_index=label)
#     print(alpha, ",",q_hat, ",", pred_set, ",", len(pred_set))

# write_to_csv("uq/q_hat/softmax.csv", alphas, q_hats, column_names=column_names)

q_hats = []
print("ROAD SCORE")
for alpha in alphas:
    q_hat = get_calibration_cutoff(model=model, 
                        calibration_dataloader=cal_loader,
                            interpreter=interpreter, 
                            cal_scoring_function=cal_road_score, alpha=alpha)
    q_hats.append(q_hat)
    print("alpha, q_hat, prediction set, prediction set length")
    pred_set = generate_prediction_set(signal, model, q_hat, interpreter, road_score, class_index=label)
    print(alpha, ",",q_hat, ",", pred_set, ",", len(pred_set))

write_to_csv("uq/q_hat/road.csv", alphas, q_hats, column_names=column_names)




# load csv into pandas dataframes 











# at some point we should also just do default interpretability on top of softmax prediction sets.
# can do that after this experiment hopefully runs successfully.


# do this for just one sample, can do across all later
# test example 


# #  testing for pytorch gradcam metrics
# from pytorch_grad_cam.metrics.road import ROADMostRelevantFirstAverage
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# targets = [ClassifierOutputTarget(label)]
# cam_metric = ROADMostRelevantFirstAverage(percentiles=[20, 40, 60, 80])
# norm_signal, cont_mask= interpreter.visualize(signal, label)
# # print(cont_mask.shape)
# cont_mask = cont_mask[np.newaxis,:]
# print(cont_mask.shape)
# # cont_mask = torch.from_numpy(cont_mask)
# scores = cam_metric(signal.cuda(), cont_mask.transpose(), targets, model)
# print(scores)
