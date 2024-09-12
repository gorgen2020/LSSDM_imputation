import argparse
import torch
import datetime
import json
import yaml
import os

from dataset_pemsbay import get_dataloader_original, get_dataloader
from vae_model_pems import VAE_pems
from utils_vae_pems import train_vae, evaluate_vae

from main_model import CSDI_pems
from utils import train, evaluate

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')

# pm25_validationindex0_20240215_214905
parser.add_argument("--modelfolder", type=str, default="")

parser.add_argument(
    "--validationindex", type=int, default=0, help="index of month used for validation (value:[0-7])"
)
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--unconditional", action="store_true", default=False)

############################################
parser.add_argument(
    "--targetstrategy", type=str, default="random", choices=["mix", "random", "block"]
)
parser.add_argument("--missing_pattern", type=str, default="point")  # block|point


#############################################################




args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["target_strategy"] = args.targetstrategy

print(json.dumps(config, indent=4))

# current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# foldername = (
#     "./save/pems_validationindex" + str(args.validationindex) + "_" + current_time + "/"
# )
#
# print('model folder:', foldername)
# os.makedirs(foldername, exist_ok=True)
# with open(foldername + "config.json", "w") as f:
#     json.dump(config, f, indent=4)
#
#
# train_loader, valid_loader, test_loader, test_train_loader, test_valid_loader,  scaler, mean_scaler = get_dataloader_original(
#     config["train_VAE"]["batch_size"], device=args.device, missing_pattern=args.missing_pattern,
#     is_interpolate=config["model"]["use_guide"], target_strategy=args.targetstrategy
# )



model_vae = VAE_pems(config, args.device).to(args.device)


if __name__ == '__main__':
    # if args.modelfolder == "":
    #     train_vae(
    #         model_vae,
    #         config["train_VAE"],
    #         train_loader,
    #         valid_loader=valid_loader,
    #         foldername=foldername,
    #     )
    # else:
    #     model_vae.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))
    #
    # evaluate_vae(
    #     model_vae,
    #     test_train_loader,
    #     test_valid_loader,
    #     test_loader,
    #     scaler=scaler,
    #     mean_scaler=mean_scaler,
    #     foldername=foldername,
    # )


    print('########################  begin diffussioh    ######################################')
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = (
            "./save/pems_validationindex" + str(args.validationindex) + "_" + current_time + "/"
    )

    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)

    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
        config["train_diffussion"]["batch_size"], device=args.device, missing_pattern=args.missing_pattern,
        is_interpolate=config["model"]["use_guide"], target_strategy=args.targetstrategy
    )

    model = CSDI_pems(config, args.device).to(args.device)

    if args.modelfolder == "":
        train(
            model,
            config["train_diffussion"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
        )
    else:
        model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

    evaluate(
        model,
        test_loader,
        nsample=args.nsample,
        scaler=scaler,
        mean_scaler=mean_scaler,
        foldername=foldername,
    )
