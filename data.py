from torch.utils.data import Dataset

class CustomDataset(Dataset):

    def __init__(self, ts_list, labels):
        self.tensor_list = ts_list
        self.labels = labels

    def __len__(self):
        return len(self.tensor_list)

    def __getitem__(self, idx):
        return self.tensor_list[idx], self.labels[idx]
