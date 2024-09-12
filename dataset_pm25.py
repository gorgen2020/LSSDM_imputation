import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
from scipy.interpolate import interp1d

class PM25_Dataset_original(Dataset):
    def __init__(self, eval_length=36, target_dim=36, mode="train", validindex=0):
        self.eval_length = eval_length
        self.target_dim = target_dim

        path = "./data/pm25/pm25_meanstd.pk"
        with open(path, "rb") as f:
            self.train_mean, self.train_std = pickle.load(f)
        if mode == "train":
            month_list = [1, 2, 4, 5, 7, 8, 10, 11]
            # 1st,4th,7th,10th months are excluded from histmask (since the months are used for creating missing patterns in test dataset)
            flag_for_histmask = [0, 1, 0, 1, 0, 1, 0, 1] 
            month_list.pop(validindex)
            flag_for_histmask.pop(validindex)
        elif mode == "valid":
            month_list = [1, 2, 4, 5, 7, 8, 10, 11]
            month_list = month_list[validindex : validindex + 1]
        elif mode == "test_train":
            month_list = [1, 2, 4, 5, 7, 8, 10, 11]
        elif mode == "test_valid":
            month_list = [1, 2, 4, 5, 7, 8, 10, 11]
            month_list = month_list[validindex : validindex + 1]
        elif mode == "test":
            month_list = [3, 6, 9, 12]
        self.month_list = month_list

        # create data for batch
        self.observed_data_interpolation = []
        #################################
        self.observed_data = []  # values (separated into each month)
        self.observed_mask = []  # masks (separated into each month)
        self.gt_mask = []  # ground-truth masks (separated into each month)
        self.index_month = []  # indicate month
        self.position_in_month = []  # indicate the start position in month (length is the same as index_month)
        self.valid_for_histmask = []  # whether the sample is used for histmask
        self.use_index = []  # to separate train/valid/test
        self.cut_length = []  # excluded from evaluation targets

        df = pd.read_csv(
            "./data/pm25/Code/STMVL/SampleData/pm25_ground.txt",
            index_col="datetime",
            parse_dates=True,
        )
        df_gt = pd.read_csv(
            "./data/pm25/Code/STMVL/SampleData/pm25_missing.txt",
            index_col="datetime",
            parse_dates=True,
        )

        # original deta preprocess
        totoal_original_mask = 0 + df.isnull().values

        content = np.array(df)
        content[8758,13] = 60.0
        content[0, 29] = 78.0

        y_hat = []
        total_mask = totoal_original_mask
        total_mask[0] = 0
        total_mask[total_mask.shape[0] - 1] = 0
        mask_seq = [i for i in range(total_mask.shape[0])]
        mask_seq = np.array(mask_seq)
        for kk in range(total_mask.shape[1]):
            x = []
            y = []
            for ii in range(total_mask.shape[0]):
                if total_mask[ii, kk] == 0:
                    x.append(mask_seq[ii])
                    y.append(content[ii, kk])
            f = interp1d(x, y)
            y_hatt = f(mask_seq)
            y_hat.append(y_hatt)
        data_seq1 = np.transpose(np.array(y_hat))

        df.loc[:] = data_seq1


        ####################### data preprocess for the data with simulated mask ########
        df_copy = df_gt.copy(deep=True)
        # replace the gt original missing data with 1D imputation
        original_missing_data = data_seq1 * totoal_original_mask
        content1 = np.array(df_gt)

        content1 = np.where(totoal_original_mask == 1,original_missing_data,content1)
        df_copy.loc[:] = content1


        simulated_mask = 0 + df_copy.isnull().values

        content = np.array(df_copy)

        content[0, 18] = 60.0
        content[0, 26] = 60.0
        content[0, 34] = 60.0
        content[0, 35] = 60.0

        content[8758, 35] = 78.0
        content[8758, 26] = 78.0
        content[8758, 27] = 78.0
        content[8758, 18] = 78.0


        y_hat = []
        total_mask = simulated_mask
        total_mask[0] = 0
        total_mask[total_mask.shape[0] - 1] = 0
        mask_seq = [i for i in range(total_mask.shape[0])]

        for kk in range(total_mask.shape[1]):
            x = []
            y = []
            for ii in range(total_mask.shape[0]):
                if total_mask[ii, kk] == 0:
                    x.append(mask_seq[ii])
                    y.append(content[ii, kk])
            f = interp1d(x, y)
            y_hatt = f(mask_seq)
            y_hat.append(y_hatt)

        ff = np.transpose(np.array(y_hat))

        df_copy.loc[:] = ff

###################################################################

        for i in range(len(month_list)):
            current_df = df[df.index.month == month_list[i]]
            current_df_gt = df_gt[df_gt.index.month == month_list[i]]
            ##############################
            current_df_interpolation = df_copy[df_copy.index.month == month_list[i]]

            ##############################
            current_length = len(current_df) - eval_length + 1
            last_index = len(self.index_month)
            self.index_month += np.array([i] * current_length).tolist()
            self.position_in_month += np.arange(current_length).tolist()
            if mode == "train":
                self.valid_for_histmask += np.array(
                    [flag_for_histmask[i]] * current_length
                ).tolist()

            # mask values for observed indices are 1
            c_mask = 1 - current_df.isnull().values
            c_gt_mask = 1 - current_df_gt.isnull().values
            c_data = (
                (current_df.fillna(0).values - self.train_mean) / self.train_std
            ) * c_mask

            c_data_interpolation = (current_df_interpolation.fillna(0).values - self.train_mean) / self.train_std

            self.observed_mask.append(c_mask)
            self.gt_mask.append(c_gt_mask)
            self.observed_data.append(c_data)

            ################
            self.observed_data_interpolation.append(c_data_interpolation)
#################################
            # print()

            if mode == "test" or mode == "test_train" or mode == "test_valid":
                n_sample = len(current_df) // eval_length
                # interval size is eval_length (missing values are imputed only once)
                c_index = np.arange(
                    last_index, last_index + eval_length * n_sample, eval_length
                )
                self.use_index += c_index.tolist()
                self.cut_length += [0] * len(c_index)
                if len(current_df) % eval_length != 0:  # avoid double-count for the last time-series
                    self.use_index += [len(self.index_month) - 1]
                    self.cut_length += [eval_length - len(current_df) % eval_length]

        if mode == "train" or mode == 'valid':
            self.use_index = np.arange(len(self.index_month))
            self.cut_length = [0] * len(self.use_index)

        # masks for 1st,4th,7th,10th months are used for creating missing patterns in test data,
        # so these months are excluded from histmask to avoid leakage
        if mode == "train":
            ind = -1
            self.index_month_histmask = []
            self.position_in_month_histmask = []

            for i in range(len(self.index_month)):
                while True:
                    ind += 1
                    if ind == len(self.index_month):
                        ind = 0
                    if self.valid_for_histmask[ind] == 1:
                        self.index_month_histmask.append(self.index_month[ind])
                        self.position_in_month_histmask.append(
                            self.position_in_month[ind]
                        )
                        break
        else:  # dummy (histmask is only used for training)
            self.index_month_histmask = self.index_month
            self.position_in_month_histmask = self.position_in_month


    def __getitem__(self, org_index):
        index = self.use_index[org_index]
        c_month = self.index_month[index]
        c_index = self.position_in_month[index]
        hist_month = self.index_month_histmask[index]
        hist_index = self.position_in_month_histmask[index]
        s = {
            "observed_data": self.observed_data[c_month][
                c_index : c_index + self.eval_length
            ],
            "observed_data_interpolation": self.observed_data_interpolation[c_month][
                             c_index: c_index + self.eval_length
            ],
            "observed_mask": self.observed_mask[c_month][
                c_index : c_index + self.eval_length
            ],
            "gt_mask": self.gt_mask[c_month][
                c_index : c_index + self.eval_length
            ],
            "hist_mask": self.observed_mask[hist_month][
                hist_index : hist_index + self.eval_length
            ],
            "timepoints": np.arange(self.eval_length),
            "cut_length": self.cut_length[org_index],
        }

        return s

    def __len__(self):
        return len(self.use_index)


class PM25_Dataset(Dataset):
    def __init__(self, eval_length=36, target_dim=36, mode="train", validindex=0):
        self.eval_length = eval_length
        self.target_dim = target_dim

        path = "./data/pm25/pm25_meanstd.pk"
        with open(path, "rb") as f:
            self.train_mean, self.train_std = pickle.load(f)
        if mode == "train":
            month_list = [1, 2, 4, 5, 7, 8, 10, 11]
            # 1st,4th,7th,10th months are excluded from histmask (since the months are used for creating missing patterns in test dataset)
            flag_for_histmask = [0, 1, 0, 1, 0, 1, 0, 1]
            month_list.pop(validindex)
            flag_for_histmask.pop(validindex)
        elif mode == "valid":
            month_list = [1, 2, 4, 5, 7, 8, 10, 11]
            month_list = month_list[validindex : validindex + 1]
        elif mode == "test_train":
            month_list = [1, 2, 4, 5, 7, 8, 10, 11]
        elif mode == "test_valid":
            month_list = [1, 2, 4, 5, 7, 8, 10, 11]
            month_list = month_list[validindex : validindex + 1]
        elif mode == "test":
            month_list = [3, 6, 9, 12]
        self.month_list = month_list

        # create data for batch
        self.observed_data_interpolation = []
        #################################
        self.observed_data = []  # values (separated into each month)
        self.observed_mask = []  # masks (separated into each month)
        self.gt_mask = []  # ground-truth masks (separated into each month)
        self.index_month = []  # indicate month
        self.position_in_month = []  # indicate the start position in month (length is the same as index_month)
        self.valid_for_histmask = []  # whether the sample is used for histmask
        self.use_index = []  # to separate train/valid/test
        self.cut_length = []  # excluded from evaluation targets

        df = pd.read_csv(
            "./data/pm25/Code/STMVL/SampleData/pm25_ground.txt",
            index_col="datetime",
            parse_dates=True,
        )
        df_gt = pd.read_csv(
            "./data/pm25/Code/STMVL/SampleData/pm25_missing.txt",
            index_col="datetime",
            parse_dates=True,
        )

        # original deta preprocess
        totoal_original_mask = 0 + df.isnull().values
        data_seq1 = np.load('restore-sequences.npz')['all_predicted_train']
        content1 = np.where(totoal_original_mask==1, data_seq1, df.values)
        df.loc[:] = content1


        # replace the gt original missing data with 1D imputation
        content1 = np.where(totoal_original_mask == 1, data_seq1, df_gt.values)
        df_gt.loc[:] = content1


        ####################### data preprocess for the data with simulated mask ########
        df_copy = df_gt.copy(deep=True)
        simulated_mask = 0 + df_copy.isnull().values
        content1 = np.where(simulated_mask==1, data_seq1, df_gt.values)
        df_copy.loc[:] = content1

###################################################################

        for i in range(len(month_list)):
            current_df = df[df.index.month == month_list[i]]
            current_df_gt = df_gt[df_gt.index.month == month_list[i]]
            ##############################
            current_df_interpolation = df_copy[df_copy.index.month == month_list[i]]

            ##############################
            current_length = len(current_df) - eval_length + 1
            last_index = len(self.index_month)
            self.index_month += np.array([i] * current_length).tolist()
            self.position_in_month += np.arange(current_length).tolist()
            if mode == "train":
                self.valid_for_histmask += np.array(
                    [flag_for_histmask[i]] * current_length
                ).tolist()

            # mask values for observed indices are 1
            c_mask = 1 - current_df.isnull().values
            c_gt_mask = 1 - current_df_gt.isnull().values
            c_data = (
                (current_df.fillna(0).values - self.train_mean) / self.train_std
            ) * c_mask

            c_data_interpolation = (current_df_interpolation.fillna(0).values - self.train_mean) / self.train_std

            self.observed_mask.append(c_mask)
            self.gt_mask.append(c_gt_mask)
            self.observed_data.append(c_data)

            ################
            self.observed_data_interpolation.append(c_data_interpolation)
#################################
            # print()

            if mode == "test" or mode == "test_train" or mode == "test_valid":
                n_sample = len(current_df) // eval_length
                # interval size is eval_length (missing values are imputed only once)
                c_index = np.arange(
                    last_index, last_index + eval_length * n_sample, eval_length
                )
                self.use_index += c_index.tolist()
                self.cut_length += [0] * len(c_index)
                if len(current_df) % eval_length != 0:  # avoid double-count for the last time-series
                    self.use_index += [len(self.index_month) - 1]
                    self.cut_length += [eval_length - len(current_df) % eval_length]

        if mode == "train" or mode == 'valid':
            self.use_index = np.arange(len(self.index_month))
            self.cut_length = [0] * len(self.use_index)

        # masks for 1st,4th,7th,10th months are used for creating missing patterns in test data,
        # so these months are excluded from histmask to avoid leakage
        if mode == "train":
            ind = -1
            self.index_month_histmask = []
            self.position_in_month_histmask = []

            for i in range(len(self.index_month)):
                while True:
                    ind += 1
                    if ind == len(self.index_month):
                        ind = 0
                    if self.valid_for_histmask[ind] == 1:
                        self.index_month_histmask.append(self.index_month[ind])
                        self.position_in_month_histmask.append(
                            self.position_in_month[ind]
                        )
                        break
        else:  # dummy (histmask is only used for training)
            self.index_month_histmask = self.index_month
            self.position_in_month_histmask = self.position_in_month


    def __getitem__(self, org_index):
        index = self.use_index[org_index]
        c_month = self.index_month[index]
        c_index = self.position_in_month[index]
        hist_month = self.index_month_histmask[index]
        hist_index = self.position_in_month_histmask[index]
        s = {
            "observed_data": self.observed_data[c_month][
                c_index : c_index + self.eval_length
            ],
            "observed_data_interpolation": self.observed_data_interpolation[c_month][
                             c_index: c_index + self.eval_length
            ],
            "observed_mask": self.observed_mask[c_month][
                c_index : c_index + self.eval_length
            ],
            "gt_mask": self.gt_mask[c_month][
                c_index : c_index + self.eval_length
            ],
            "hist_mask": self.observed_mask[hist_month][
                hist_index : hist_index + self.eval_length
            ],
            "timepoints": np.arange(self.eval_length),
            "cut_length": self.cut_length[org_index],
        }

        return s

    def __len__(self):
        return len(self.use_index)

def get_dataloader_original(batch_size, device, validindex=0):
    dataset = PM25_Dataset_original(mode="train", validindex=validindex)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=1, shuffle=True
    )

    dataset_valid = PM25_Dataset_original(mode="valid", validindex=validindex)
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size, num_workers=1, shuffle=False
    )

    ###########################
    dataset_test = PM25_Dataset_original(mode="test", validindex=validindex)
    test_loader = DataLoader(
        dataset_test, batch_size=batch_size, num_workers=1, shuffle=False
    )

    dataset_test_train = PM25_Dataset_original(mode="test_train", validindex=validindex)
    test_train_loader = DataLoader(
        dataset_test_train, batch_size=batch_size, num_workers=1, shuffle=False
    )

    dataset_test_valid = PM25_Dataset_original(mode="test_valid", validindex=validindex)
    test_valid_loader = DataLoader(
        dataset_test_valid, batch_size=batch_size, num_workers=1, shuffle=False
    )
    ###############################################

    scaler = torch.from_numpy(dataset.train_std).to(device).float()
    mean_scaler = torch.from_numpy(dataset.train_mean).to(device).float()

    return train_loader, valid_loader, test_loader, test_train_loader, test_valid_loader, scaler, mean_scaler


def get_dataloader(batch_size, device, validindex=0):
    dataset = PM25_Dataset(mode="train", validindex=validindex)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=1, shuffle=True
    )

    dataset_valid = PM25_Dataset(mode="valid", validindex=validindex)
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size, num_workers=1, shuffle=False
    )

    ###########################
    dataset_test = PM25_Dataset(mode="test", validindex=validindex)
    test_loader = DataLoader(
        dataset_test, batch_size=batch_size, num_workers=1, shuffle=False
    )

    dataset_test_train = PM25_Dataset(mode="test_train", validindex=validindex)
    test_train_loader = DataLoader(
        dataset_test_train, batch_size=batch_size, num_workers=1, shuffle=False
    )

    dataset_test_valid = PM25_Dataset(mode="test_valid", validindex=validindex)
    test_valid_loader = DataLoader(
        dataset_test_valid, batch_size=batch_size, num_workers=1, shuffle=False
    )
    ###############################################

    scaler = torch.from_numpy(dataset.train_std).to(device).float()
    mean_scaler = torch.from_numpy(dataset.train_mean).to(device).float()

    return train_loader, valid_loader, test_loader, test_train_loader, test_valid_loader, scaler, mean_scaler
