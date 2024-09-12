import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pandas as pd
import pickle

def train_vae(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=20,
    foldername="",
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
                if batch_no >= config["itr_per_epoch"]:
                    break

            lr_scheduler.step()
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )

    if foldername != "":
        torch.save(model.state_dict(), output_path)


def evaluate_vae(model, train_loader, valid_loader, test_loader, scaler=1, mean_scaler=0, foldername="", missingrate=0.1):
    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target_train = []
        all_observed_point_train = []
        all_observed_time_train = []
        all_evalpoint_train = []
        all_generated_samples_train = []

        all_target_valid = []
        all_observed_point_valid = []
        all_observed_time_valid = []
        all_evalpoint_valid = []
        all_generated_samples_valid = []


        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []

        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                output = model.evaluate(train_batch)
                c_target, eval_points, observed_points, observed_time, predicted_value = output

                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                predicted_value = predicted_value.permute(0, 2, 1)
                ####################
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                #####################################################################
                all_target_train.append(c_target * scaler + mean_scaler)
                all_evalpoint_train.append(eval_points)
                all_observed_point_train.append(observed_points)
                all_observed_time_train.append(observed_time)
                all_generated_samples_train.append(predicted_value * scaler + mean_scaler)



        with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, valid_batch in enumerate(it, start=1):
                output = model.evaluate(valid_batch)
                c_target, eval_points, observed_points, observed_time, predicted_value = output
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                predicted_value = predicted_value.permute(0, 2, 1)
                ####################
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)
                #####################################################################
                all_target_valid.append(c_target * scaler + mean_scaler)
                all_evalpoint_valid.append(eval_points)
                all_observed_point_valid.append(observed_points)
                all_observed_time_valid.append(observed_time)
                all_generated_samples_valid.append(predicted_value * scaler + mean_scaler)


        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch)

                c_target, eval_points, observed_points, observed_time, predicted_value = output

                c_target = c_target.permute(0, 2, 1)  # (B,L,K)

                predicted_value = predicted_value.permute(0, 2, 1)

                ##################################
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                #####################################################################
                all_target.append(c_target * scaler + mean_scaler)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(predicted_value * scaler + mean_scaler)

                mse_current = (
                                      ((predicted_value - c_target) * eval_points) ** 2
                              ) * (scaler ** 2)
                mae_current = (
                                  torch.abs((predicted_value - c_target) * eval_points)
                              ) * scaler

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )



        #########################################################################
        all_target_train = torch.cat(all_target_train, dim=0)
        all_evalpoint_train = torch.cat(all_evalpoint_train, dim=0)
        all_generated_samples_train = torch.cat(all_generated_samples_train, dim=0)

        all_target_train = all_target_train.cpu().numpy()

        all_generated_samples_train = all_generated_samples_train.cpu().numpy()

        all_evalpoint_train = all_evalpoint_train.cpu().numpy()


        np.savez('train_result.npz', all_target_train=all_target_train,
                 all_generated_samples_train=all_generated_samples_train,
                 all_evalpoint_train=all_evalpoint_train)


        ##########################################################################

        all_target_valid = torch.cat(all_target_valid, dim=0)
        all_evalpoint_valid = torch.cat(all_evalpoint_valid, dim=0)
        all_generated_samples_valid = torch.cat(all_generated_samples_valid, dim=0)
        all_target_valid = all_target_valid.cpu().numpy()
        all_generated_samples_valid = all_generated_samples_valid.cpu().numpy()
        all_evalpoint_valid = all_evalpoint_valid.cpu().numpy()
        np.savez('valid_result.npz', all_target_valid=all_target_valid,
                 all_generated_samples_valid=all_generated_samples_valid,
                 all_evalpoint_valid=all_evalpoint_valid)

        #######################################################################

        all_target = torch.cat(all_target, dim=0)
        all_evalpoint = torch.cat(all_evalpoint, dim=0)
        all_generated_samples = torch.cat(all_generated_samples, dim=0)

        all_target = all_target.cpu().numpy()

        all_generated_samples = all_generated_samples.cpu().numpy()

        all_evalpoint = all_evalpoint.cpu().numpy()



        np.savez('test_result.npz', all_target=all_target, all_generated_samples=all_generated_samples,
                 all_evalpoint=all_evalpoint)


        path = ("./data/physio_missing0.1" + "_seed1" + ".pk")
        with open(path, "rb") as f:
            observed_values, observed_masks, gt_masks = pickle.load(f)

        seed =1
        nfold=0

        indlist = np.arange(len(observed_values))

        np.random.seed(seed)
        np.random.shuffle(indlist)

        # 5-fold test
        start = (int)(nfold * 0.2 * len(observed_values))
        end = (int)((nfold + 1) * 0.2 * len(observed_values))
        test_index = indlist[start:end]
        remain_index = np.delete(indlist, np.arange(start, end))

        np.random.seed(seed)
        np.random.shuffle(remain_index)
        num_train = (int)(len(observed_values) * 0.7)
        train_index = remain_index[:num_train]
        valid_index = remain_index[num_train:]



        ###################################################
        original_length = len(observed_values)

        # Array to store all predictions in the original order
        reversed_all_targets = np.zeros_like(observed_values)

        # Place the predictions back in their original positions
        reversed_all_targets[train_index] = all_generated_samples_train
        reversed_all_targets[valid_index] = all_generated_samples_valid
        reversed_all_targets[test_index] = all_generated_samples

        np.savez('restore-sequences.npz', all_predicted_train=reversed_all_targets)

