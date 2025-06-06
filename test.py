import os
import time
import datetime
import numpy as np
import torch
from utils.utils import print_and_save, epoch_time,my_seeding
from network.model import DecoSeg
from utils.metrics import DiceBCELoss
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
from utils.run_train_vail import train, evaluate
import argparse
from utils.utils import calculate_params_flops
from dataset.loader import get_loader

DW_POS=1
DS_POS=2
C_POS=3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        default="Monu_Seg",
        help="input datasets name including ISIC2018, PH2, Kvasir, BUSI, COVID_19,CVC_ClinkDB,Monu_Seg",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=5,
        help="gpu_id:",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default="12",
        help="input batch_size",
    )
    parser.add_argument(
        "--imagesize",
        type=int, 
        default=256,
        help="input image resolution.",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="log",
        help="input log folder: ./log",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="the checkpoint path of last model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random configure:",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=400,
        help="end epoch",
    )
    parser.add_argument(
        "--in_channels",
        type=list,
        default=[64, 256, 512],
        help="in_channels",
    )
    parser.add_argument(
        "--pos",
        type=int,
        default=DS_POS,
        help="pos",
    )  
    parser.add_argument(
        "--H_kernel",
        type=list,
        default=[(3,3),(5,1),(1,5)],
        help="",
    )
    parser.add_argument(
        "--L_kernel",
        type=list,
        default=[2,4],
        help="",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="out_channels",
    )
    parser.add_argument(
        "--esp",
        type=int,
        default=200,
        help="out_channels",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default='train',
        help="mode",
    )
    return parser.parse_args()



if __name__ == "__main__":
    TEST,TRAIN = 1,2
    
    args = parse_args()
    dataset_name = args.datasets

    seed=args.seed
    my_seeding(seed)
    image_size = args.imagesize
    size = (image_size, image_size)
    batch_size = args.batchsize
    num_epochs = args.epoch
    lr = args.lr
    lr_backbone = args.lr
    early_stopping_patience = args.esp

    pretrained_backbone = args.checkpoint
    resume_path = args.checkpoint

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = f"{dataset_name}_{current_time}"

    save_dir = os.path.join(args.log, folder_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_log_path = os.path.join(save_dir, "train_log.txt")
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")

    train_log = open(train_log_path, "w")
    train_log.write("\n")
    train_log.close()

    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)
    print("")

    hyperparameters_str = f"Image Size: {image_size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    hyperparameters_str += f"Early Stopping Patience: {early_stopping_patience}\n"
    hyperparameters_str += f"Seed: {seed}\n"
    print_and_save(train_log_path, hyperparameters_str)

    
    train_loader, valid_loader = get_loader(dataset_name, batch_size, image_size, train_log_path)
    
    device = torch.device(f'cuda:{args.gpu}')
    
    model = DecoSeg(in_channels=args.in_channels,pos=args.pos,l_kernel=args.H_kernel[0],v_kernel=args.H_kernel[1],h_kernel=args.H_kernel[2],p_kernel=args.L_kernel)
    if pretrained_backbone:
        saved_weights = torch.load(pretrained_backbone)
        for name, param in model.named_parameters():
            if name.startswith('backbone.layer0') or name.startswith('backbone.layer1') or name.startswith('backbone.layer2') or name.startswith(
                    'backbone.layer3'):
                param.data = saved_weights[name]

    if resume_path:
        checkpoint = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(checkpoint)

    model = model.to(device)
    
    param_groups = [
        {'params': [], 'lr': lr_backbone},
        {'params': [], 'lr': lr}
    ]
  
    for name, param in model.named_parameters():
        if name.startswith('backbone.layer0') or name.startswith('backbone.layer1') or name.startswith('backbone.layer2') or name.startswith('backbone.layer3'):
            param_groups[0]['params'].append(param)
        else:
            param_groups[1]['params'].append(param)

    assert len(param_groups[0]['params']) > 0, "Layer group is empty!"
    assert len(param_groups[1]['params']) > 0, "Rest group is empty!"

    optimizer = torch.optim.Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=30, verbose=True)
    loss_fn = DiceBCELoss()
    loss_name = "BCE Dice Loss"
    data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    data_str = f"Number of parameters: {num_params / 1000000}M\n"
    print_and_save(train_log_path, data_str)


    best_valid_metrics = 0.0
    early_stopping_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()
        valid_loss, valid_metrics = evaluate(model, valid_loader, loss_fn, device)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        # data_str += f"\tTrain Loss: {train_loss:.4f} - miou:{train_metrics[0]},dsc:{train_metrics[1]},acc:{train_metrics[2]},sen:{train_metrics[3]},spe:{train_metrics[4]},pre:{train_metrics[5]},rec:{train_metrics[6]},fb:{train_metrics[7]},em:{train_metrics[8]}\n"
        data_str += f"\t Val. Loss: {valid_loss:.4f} - miou:{valid_metrics[0]},dsc:{valid_metrics[1]},acc:{valid_metrics[2]},sen:{valid_metrics[3]},spe:{valid_metrics[4]},pre:{valid_metrics[5]},rec:{valid_metrics[6]},fb:{valid_metrics[7]},em:{valid_metrics[8]}\n"
        print_and_save(train_log_path, data_str)

        