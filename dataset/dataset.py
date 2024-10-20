import pandas as pd
import pickle
from torch.utils.data import Dataset

class EFL_dataset(Dataset):
    def __init__(self, dataset_path) -> None:
        with open(dataset_path, 'r') as f:
            dataframe = pd.read_csv(f)

        self.dataset = dataframe["Essay"].tolist()
        self.label = dataframe["Overall"].tolist()

    def __getitem__(self, index):
        return self.dataset[index], self.label[index]

    def __len__(self):
        return len(self.dataset)

class AsapDataset(Dataset):
    def __init__(self, dataset_path, train : bool = True) -> None:
        self.dataset = list()
        self.label = list()
        if train:
            with open(f"{dataset_path[:-10]}{(int(dataset_path[-10]) % 8) + 1}/train.pk", 'rb') as f:
                dataset = pickle.load(f)
                self.dataset = [data["content_text"] for data in dataset if data["prompt_id"] == dataset_path[-10]]
                self.label = [int(data["score"]) for data in dataset if data["prompt_id"] == dataset_path[-10]]
                self.max_label = max(self.label)
                print(self.max_label)
                self.label = [label / self.max_label for label in self.label]
        else:
            with open(f"{dataset_path[:-9]}{(int(dataset_path[-9]) % 8) + 1}/dev.pk", 'rb') as f:
                dataset = pickle.load(f)
                self.dataset = [data["content_text"] for data in dataset if data["prompt_id"] == dataset_path[-9]]
                self.label = [int(data["score"]) for data in dataset if data["prompt_id"] == dataset_path[-9]]

    def __getitem__(self, index):
        return self.dataset[index], self.label[index]

    def __len__(self):
        return len(self.dataset)
    
class AsapMixedDataset(Dataset):
    def __init__(self, dataset_path, train : bool = True) -> None:
        self.train = train
        self.dataset = list()
        self.label = list()
        self.dataset_label = list()
        if train:
            for id in range(1, 9):
                with open(f"/scratch1/jt/test/data/{id % 8 + 1}/train.pk", 'rb') as f:
                    dataset = pickle.load(f)
                    self.dataset += [f"[Prompt {str(id)}] " + data["content_text"] for data in dataset if data["prompt_id"] == str(id)]
                    label_list = [int(data["score"]) for data in dataset if data["prompt_id"] == str(id)]
                    max_label = max(label_list)
                    self.label += [int((label) / max_label * 60) for label in label_list]
                    self.dataset_label += [id - 1] * len(label_list)
            # print(self.dataset)
            # exit()
            self.max_label = max(self.label)
        else:
            with open(f"{dataset_path[:-9]}{(int(dataset_path[-9]) % 8) + 1}/dev.pk", 'rb') as f:
                dataset = pickle.load(f)
                self.dataset = [f"[Prompt {dataset_path[-9]}] " + data["content_text"] for data in dataset if data["prompt_id"] == dataset_path[-9]]
                self.label = [int(data["score"]) for data in dataset if data["prompt_id"] == dataset_path[-9]]
                self.max_label = max(self.label)
    def __getitem__(self, index):
        if self.train:
            return self.dataset[index], self.label[index], self.dataset_label[index]
        else :
            return self.dataset[index], self.label[index]

    def __len__(self):
        return len(self.dataset)
    
class AsapMixedDatasetForRegression(Dataset):
    def __init__(self, dataset_path, train : bool = True, prompt_id = None) -> None:
        self.train = train
        self.dataset = list()
        self.label = list()
        self.dataset_label = list()
        self.label_scaler = {"1" : 10, "2" : 5, "3" : 3, "4" : 3, "5" : 4, "6" : 4, "7" : 30, "8" : 60}
        self.label_min = {"1" : 2, "2" : 1, "3" : 0, "4" : 0, "5" : 0, "6" : 0, "7" : 0, "8" : 0}
        if train:
            # for id in range(1, 9):
            #     with open(f"/scratch1/jt/test/data/{id % 8 + 1}/train.pk", 'rb') as f:
            #         dataset = pickle.load(f)
            #         self.dataset += [f"[Prompt {str(id)}] " + data["content_text"] for data in dataset if data["prompt_id"] == str(id)]
            #         label_list = [int(data["score"]) for data in dataset if data["prompt_id"] == str(id)]
            #         max_label = max(label_list)
            #         self.label += [label / max_label for label in label_list]
            #         self.dataset_label += [id - 1] * len(label_list)
            # print(len(self.dataset_label))
            with open(dataset_path, 'rb') as f:
                dataset = pickle.load(f)
                # dataset = [data for data in dataset if str(data['prompt_id']) == "1" or str(data['prompt_id']) == "8"]
                print(len(dataset))
                self.dataset += [f"[Prompt {data['prompt_id']}] " + data["content_text"] for data in dataset]
                # self.dataset += [data["content_text"] for data in dataset]
                self.label = [(int(data["score"]) - self.label_min[str(data["prompt_id"])]) / self.label_scaler[str(data["prompt_id"])] for data in dataset]
                self.dataset_label += [int(data["prompt_id"]) - 1 for data in dataset]
        else:
            with open(dataset_path, 'rb') as f:
                dataset = pickle.load(f)
                self.dataset = [f"[Prompt {data['prompt_id']}] " + data["content_text"] for data in dataset]
                # self.dataset = [data["content_text"] for data in dataset]
                self.label = [int(data["score"]) for data in dataset]
                
    def __getitem__(self, index):
        if self.train:
            return self.dataset[index], self.label[index], self.dataset_label[index]
        else :
            return self.dataset[index], self.label[index]

    def __len__(self):
        return len(self.dataset)