import torch
from utils import *


class IIICDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y 
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return torch.FloatTensor(self.X[index]), np.argmax(self.Y[index])

def prepare_TUEV_dataloader(sampling_rate=200, batch_size = 512, num_workers=32):
    # set random seed
    seed = 4523
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = "/srv/local/data/TUH/tuh_eeg_events/v2.0.0/edf"

    train_files = os.listdir(os.path.join(root, "processed_train"))
    train_sub = list(set([f.split("_")[0] for f in train_files]))
    print("train sub", len(train_sub))
    test_files = os.listdir(os.path.join(root, "processed_eval"))

    val_sub = np.random.choice(train_sub, size=int(
        len(train_sub) * 0.1), replace=False)
    train_sub = list(set(train_sub) - set(val_sub))
    val_files = [f for f in train_files if f.split("_")[0] in val_sub]
    train_files = [f for f in train_files if f.split("_")[0] in train_sub]

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        TUEVLoader(
            os.path.join(
                root, "processed_train"), train_files, sampling_rate=sampling_rate
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        TUEVLoader(
            os.path.join(
                root, "processed_eval"), test_files, sampling_rate=sampling_rate
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        TUEVLoader(
            os.path.join(
                root, "processed_train"), val_files, sampling_rate=sampling_rate
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )
    print(len(train_files), len(val_files), len(test_files))
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader

def prepare_TUEV_cal_dataloader(sampling_rate=200, batch_size = 512, num_workers=32):
    # set random seed
    seed = 4523
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = "/srv/local/data/TUH/tuh_eeg_events/v2.0.0/edf"

    train_files = os.listdir(os.path.join(root, "processed_train"))
    train_sub = list(set([f.split("_")[0] for f in train_files]))
    print("train sub", len(train_sub))
    test_files = os.listdir(os.path.join(root, "processed_eval"))

    val_sub = np.random.choice(train_sub, size=int(
        len(train_sub) * 0.1), replace=False)
    train_sub = list(set(train_sub) - set(val_sub))
    val_files = [f for f in train_files if f.split("_")[0] in val_sub]
    train_files = [f for f in train_files if f.split("_")[0] in train_sub]

    # prepare training and test data loader

    train = TUEVLoader(
            os.path.join(
                root, "processed_train"), train_files, sampling_rate=sampling_rate
            )
    # use 20 percent of training for calibration
    n = len(train)
    split_train = int(0.8 * n)
    split_cal = n - split_train 
    train, cal = torch.utils.data.random_split(train, [split_train, split_cal], generator=torch.Generator().manual_seed(42))
    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=True,
    )
    cal_loader =torch.utils.data.DataLoader(
        cal,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        TUEVLoader(
            os.path.join(
                root, "processed_eval"), test_files, sampling_rate=sampling_rate
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        TUEVLoader(
            os.path.join(
                root, "processed_train"), val_files, sampling_rate=sampling_rate
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )
    print(len(train_files), len(val_files), len(test_files))
    print(len(train_loader), len(val_loader), len(test_loader), len(cal_loader))
    return train_loader, test_loader, val_loader, cal_loader


def prepare_TUEV_cal_dataloader(sampling_rate=200, batch_size = 512, num_workers=32):
    # set random seed
    seed = 4523
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = "/srv/local/data/TUH/tuh_eeg_events/v2.0.0/edf"

    train_files = os.listdir(os.path.join(root, "processed_train"))
    train_sub = list(set([f.split("_")[0] for f in train_files]))
    print("train sub", len(train_sub))
    test_files = os.listdir(os.path.join(root, "processed_eval"))

    val_sub = np.random.choice(train_sub, size=int(
        len(train_sub) * 0.1), replace=False)
    train_sub = list(set(train_sub) - set(val_sub))
    val_files = [f for f in train_files if f.split("_")[0] in val_sub]
    train_files = [f for f in train_files if f.split("_")[0] in train_sub]

    # prepare training and test data loader

    train = TUEVLoader(
            os.path.join(
                root, "processed_train"), train_files, sampling_rate=sampling_rate
            )
    # use 20 percent of training for calibration
    n = len(train)
    split_train = int(0.8 * n)
    split_cal = n - split_train 
    train, cal = torch.utils.data.random_split(train, [split_train, split_cal], generator=torch.Generator().manual_seed(42))
    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=True,
    )
    cal_loader =torch.utils.data.DataLoader(
        cal,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        TUEVLoader(
            os.path.join(
                root, "processed_eval"), test_files, sampling_rate=sampling_rate
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        TUEVLoader(
            os.path.join(
                root, "processed_train"), val_files, sampling_rate=sampling_rate
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )
    print(len(train_files), len(val_files), len(test_files))
    print(len(train_loader), len(val_loader), len(test_loader), len(cal_loader))
    return train_loader, test_loader, val_loader, cal_loader




def prepare_TUEV_datasets(sampling_rate=200):

    # set random seed
    seed = 4523
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = "/srv/local/data/TUH/tuh_eeg_events/v2.0.0/edf"

    train_files = os.listdir(os.path.join(root, "processed_train"))
    train_sub = list(set([f.split("_")[0] for f in train_files]))
    print("train sub", len(train_sub))
    test_files = os.listdir(os.path.join(root, "processed_eval"))

    val_sub = np.random.choice(train_sub, size=int(
        len(train_sub) * 0.1), replace=False)
    train_sub = list(set(train_sub) - set(val_sub))
    val_files = [f for f in train_files if f.split("_")[0] in val_sub]
    train_files = [f for f in train_files if f.split("_")[0] in train_sub]

    # prepare training and test data loader
    train = TUEVLoader(
            os.path.join(
                root, "processed_train"), train_files, sampling_rate=sampling_rate
        )
    test =  TUEVLoader(
            os.path.join(
                root, "processed_eval"), test_files, sampling_rate=sampling_rate
        ) 

    val =  TUEVLoader(
            os.path.join(
                root, "processed_train"), val_files, sampling_rate=sampling_rate
        )
    
    print(len(train_files), len(val_files), len(test_files))
    print(len(train), len(val), len(test))
    return train, test, val


def prepare_TUAB_dataloader(batch_size, num_workers , sampling_rate):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = "/srv/local/data/TUH/tuh3/tuh_eeg_abnormal/v3.0.0/edf/processed"

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    # train_files = train_files[:100000]
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "train"),
                   train_files, sampling_rate),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "test"), test_files, sampling_rate),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "val"), val_files, sampling_rate),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader


def prepare_TUAB_datasets(sampling_rate):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = "/srv/local/data/TUH/tuh3/tuh_eeg_abnormal/v3.0.0/edf/processed"

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    # train_files = train_files[:100000]
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_loader = TUABLoader(os.path.join(root, "train"), train_files, sampling_rate)
    test_loader = TUABLoader(os.path.join(root, "test"), test_files, sampling_rate)
    val_loader = TUABLoader(os.path.join(root, "val"), val_files, sampling_rate)
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader




def prepare_IIIC_cal_dataloader(batch_size = 512, num_workers=32, drop_last=False):
    # set random seed
    seed = 4523
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = "/srv/local/data/IIIC_data"

    key_label_path = os.path.join(root, "data_1109")
    data_path = os.path.join(root, "17_data_EEG_1115")

    train_key = np.load(os.path.join(key_label_path, "training_key.npy"))
    train_Y = np.load(os.path.join(key_label_path, "training_Y.npy"))
    test_key = np.load(os.path.join(key_label_path, "training_key.npy"))
    test_Y = np.load(os.path.join(key_label_path, "test_Y.npy"))

    train_X = np.load(os.path.join(data_path, "training_X.npy"))
    test_X = np.load((os.path.join(data_path, "test_X.npy")))
    train = IIICDataset(train_X, train_Y)
    test = IIICDataset(test_X, test_Y)
    n = len(train)
    split_train = int(0.8 * n)
    split_cal = int((n - split_train)/ 2) 
    splt_val = int((n - split_train)/ 2) 

    train, cal, val = torch.utils.data.random_split(train, [split_train, split_cal, splt_val], generator=torch.Generator().manual_seed(42))
    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=num_workers,
        persistent_workers=True,
    )
    cal_loader =torch.utils.data.DataLoader(
        cal,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        drop_last= drop_last, 
        num_workers=num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )
    print(len(train), len(val), len(test), len(cal))
    print(len(train_loader), len(val_loader), len(test_loader), len(cal_loader))
    return train_loader, test_loader, val_loader, cal_loader