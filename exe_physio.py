import argparse
import torch
import datetime
import json
import yaml
import os

from main_model import CSDI_Physio
from dataset_physio import get_dataloader_original, get_dataloader
from utils import train, evaluate

from vae_model_physio import VAE_Physio
from utils_vae_physio import train_vae, evaluate_vae



parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.1)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/physio_fold" + str(args.nfold) + "_" + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader, test_train_loader, test_valid_loader = get_dataloader_original(
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train_VAE"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
)


model_vae = VAE_Physio(config, args.device).to(args.device)


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

    # evaluate_vae(
    #     model_vae, test_loader, scaler=1, foldername=foldername)

    evaluate_vae(
        model_vae,
        test_train_loader,
        test_valid_loader,
        test_loader,
        scaler=1,
        mean_scaler=0,
        foldername=foldername,
        missingrate=config["model"]["test_missing_ratio"]
    )


    print('########################  begin diffussioh    ######################################')
#############################################

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = "./save/physio_fold" + str(args.nfold) + "_" + current_time + "/"
    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)
    #################################################
    train_loader1, valid_loader1, test_loader1 = get_dataloader(
        seed=args.seed,
        nfold=args.nfold,
        batch_size=config["train_diffussion"]["batch_size"],
        missing_ratio=config["model"]["test_missing_ratio"],
    )

    model = CSDI_Physio(config, args.device).to(args.device)
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

    evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)