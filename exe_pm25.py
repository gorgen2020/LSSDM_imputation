import argparse
import torch
import datetime
import json
import yaml
import os

from dataset_pm25 import get_dataloader_original, get_dataloader
from vae_model_pm25 import VAE_PM25
from utils_vae_pm25 import train_vae, evaluate_vae

from main_model import CSDI_PM25
from utils import train, evaluate

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')

# pm25_validationindex0_20240215_214905
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument(
    "--targetstrategy", type=str, default="mix", choices=["mix", "random", "historical"]
)
parser.add_argument(
    "--validationindex", type=int, default=0, help="index of month used for validation (value:[0-7])"
)
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--unconditional", action="store_true",default=False)
args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["target_strategy"] = args.targetstrategy

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
foldername = (
    "./save_vae/pm25_validationindex" + str(args.validationindex) + "_" + current_time + "/"
)

print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)
########################################################
train_loader, valid_loader, test_loader, test_train_loader, test_valid_loader, scaler, mean_scaler = get_dataloader_original(
    config["train_VAE"]["batch_size"], device=args.device, validindex=args.validationindex
)

model_vae = VAE_PM25(config, args.device).to(args.device)


if __name__ == '__main__':

    if args.modelfolder == "":
        train_vae(
            model_vae,
            config["train_VAE"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
        )
    else:
        model_vae.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

    evaluate_vae(
        model_vae,
        test_train_loader,
        test_valid_loader,
        test_loader,
        scaler=scaler,
        mean_scaler=mean_scaler,
        foldername=foldername,
    )

    print('########################  begin diffussioh    ######################################')
#############################################

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = (
        "./save/pm25_validationindex" + str(args.validationindex) + "_" + current_time + "/"
    )

    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)
    #################################################
    train_loader1, valid_loader1, test_loader1, test_train_loader1, test_valid_loader1, scaler1, mean_scaler1 = get_dataloader(
        config["train_diffussion"]["batch_size"], device=args.device, validindex=args.validationindex
    )
    model = CSDI_PM25(config, args.device).to(args.device)
    ############################################################
###################################################################
    if args.modelfolder == "":
        train(
            model,
            config["train_diffussion"],
            train_loader1,
            valid_loader=valid_loader1,
            foldername=foldername,
        )
    else:
        model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

    evaluate(
        model,
        test_loader1,
        nsample=args.nsample,
        scaler=scaler1,
        mean_scaler=mean_scaler1,
        foldername=foldername,
    )