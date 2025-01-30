# -*- coding: utf-8 -*-
import argparse
import os
import shutil
import time
import yaml
import numpy as np
import torch
import torch.optim as optim
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from config.load_config import load_yaml, DotDict
from data.dataset import SynthTextDataSet, CustomDataset
from loss.mseloss import Maploss_v2, Maploss_v3
from model.craft import CRAFT
from eval import main_eval
from metrics.eval_det_iou import DetectionIoUEvaluator
from utils.util import copyStateDict


class Trainer(object):
    def __init__(self, config, gpu, mode):
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu}")
            self.gpu = gpu
        else:
            self.device = torch.device("cpu")
            self.gpu = None

        self.config = config
        self.mode = mode
        self.net_param = self.get_load_param(self.device)

    def get_synth_loader(self):
        dataset = SynthTextDataSet(
            output_size=self.config.train.data.output_size,
            data_dir=self.config.train.synth_data_dir,
            saved_gt_dir=None,
            mean=self.config.train.data.mean,
            variance=self.config.train.data.variance,
            gauss_init_size=self.config.train.data.gauss_init_size,
            gauss_sigma=self.config.train.data.gauss_sigma,
            enlarge_region=self.config.train.data.enlarge_region,
            enlarge_affinity=self.config.train.data.enlarge_affinity,
            aug=self.config.train.data.syn_aug,
            vis_test_dir=self.config.vis_test_dir,
            vis_opt=self.config.train.data.vis_opt,
            sample=self.config.train.data.syn_sample,
        )

        syn_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.train.batch_size // self.config.train.synth_ratio,
            shuffle=False,
            num_workers=self.config.train.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return syn_loader

    def get_custom_dataset(self):
        custom_dataset = CustomDataset(
            output_size=self.config.train.data.output_size,
            data_dir=self.config.data_root_dir,
            saved_gt_dir=None,
            mean=self.config.train.data.mean,
            variance=self.config.train.data.variance,
            gauss_init_size=self.config.train.data.gauss_init_size,
            gauss_sigma=self.config.train.data.gauss_sigma,
            enlarge_region=self.config.train.data.enlarge_region,
            enlarge_affinity=self.config.train.data.enlarge_affinity,
            watershed_param=self.config.train.data.watershed,
            aug=self.config.train.data.custom_aug,
            vis_test_dir=self.config.vis_test_dir,
            sample=self.config.train.data.custom_sample,
            vis_opt=self.config.train.data.vis_opt,
            pseudo_vis_opt=self.config.train.data.pseudo_vis_opt,
            do_not_care_label=self.config.train.data.do_not_care_label,
        )

        return custom_dataset

    def get_load_param(self, device):
        if self.config.train.ckpt_path is not None:
            map_location = device
            param = torch.load(self.config.train.ckpt_path, map_location=map_location)
        else:
            param = None

        return param

    def adjust_learning_rate(self, optimizer, gamma, step, lr):
        lr = lr * (gamma**step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return param_group["lr"]

    def get_loss(self):
        if self.config.train.loss == 2:
            criterion = Maploss_v2()
        elif self.config.train.loss == 3:
            criterion = Maploss_v3()
        else:
            raise Exception("Undefined loss")
        return criterion

    def iou_eval(self, dataset, train_step, buffer, model):
        test_config = DotDict(self.config.test[dataset])

        val_result_dir = os.path.join(
            self.config.results_dir, "{}/{}".format(dataset + "_iou", str(train_step))
        )

        evaluator = DetectionIoUEvaluator()

        metrics = main_eval(
            None,
            self.config.train.backbone,
            test_config,
            evaluator,
            val_result_dir,
            buffer,
            model,
            self.mode,
        )
        if self.gpu == 0 and self.config.wandb_opt:
            wandb.log(
                {
                    "{} iou Recall".format(dataset): np.round(metrics["recall"], 3),
                    "{} iou Precision".format(dataset): np.round(
                        metrics["precision"], 3
                    ),
                    "{} iou F1-score".format(dataset): np.round(metrics["hmean"], 3),
                }
            )

    def iou_train(
        self, pred_scores: torch.Tensor, gt_scores: torch.Tensor, threshold=0.5
    ):
        """
        Compute Intersection over Union (IoU) for text detection.
        Args:
            pred_scores: Model-predicted region or affinity scores.
            gt_scores: Ground truth region or affinity scores.
        """
        intersection = (
            torch.logical_and(pred_scores > threshold, gt_scores > threshold)
            .sum()
            .item()
        )
        union = (
            torch.logical_or(pred_scores > threshold, gt_scores > threshold)
            .sum()
            .item()
        )

        if union == 0:
            return (
                1.0 if intersection == 0 else 0.0
            )  # Handle edge case where no text is present

        return intersection / union

    def plot_learning_curves(self, train_loss, train_iou):
        fig, ax1 = plt.subplots()

        color = "tab:red"
        ax1.set_xlabel("Training Steps")
        ax1.set_ylabel("Loss", color=color)
        ax1.plot(train_loss, color=color)
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()
        color = "tab:blue"
        ax2.set_ylabel("IoU", color=color)
        ax2.plot(train_iou, color=color)
        ax2.tick_params(axis="y", labelcolor=color)

        fig.tight_layout()
        plt.show()

    def train(self, buffer_dict):
        # Check if CUDA is available, else use CPU
        device = torch.device(
            f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu"
        )

        # MODEL -------------------------------------------------------------------------------------------------------#
        # SUPERVISION model
        if self.config.mode == "weak_supervision":
            if self.config.train.backbone == "vgg":
                supervision_model = CRAFT(pretrained=False, amp=self.config.train.amp)
            else:
                raise Exception("Undefined architecture")

            if self.config.train.ckpt_path is not None:
                supervision_param = self.get_load_param(device)
                supervision_model.load_state_dict(
                    copyStateDict(supervision_param["craft"])
                )

            supervision_model = supervision_model.to(device)
            print(f"Supervision model loading on: {device}")
        else:
            supervision_model = None

        # TRAIN model
        if self.config.train.backbone == "vgg":
            craft = CRAFT(pretrained=True, amp=self.config.train.amp)
        else:
            raise Exception("Undefined architecture")

        if self.config.train.ckpt_path is not None:
            craft.load_state_dict(copyStateDict(self.net_param["craft"]))

        craft = craft.to(device)
        craft = torch.nn.DataParallel(craft)

        # Disable cudnn.benchmark if using CPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        # DATASET -----------------------------------------------------------------------------------------------------#

        if self.config.train.use_synthtext:
            trn_syn_loader = self.get_synth_loader()
            batch_syn = iter(trn_syn_loader)

        if self.config.train.real_dataset == "custom":
            trn_real_dataset = self.get_custom_dataset()
        else:
            raise Exception("Undefined dataset")

        if self.config.mode == "weak_supervision":
            trn_real_dataset.update_model(supervision_model)
            trn_real_dataset.update_device(device)

        trn_real_loader = torch.utils.data.DataLoader(
            trn_real_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=self.config.train.num_workers,
            drop_last=False,
        )

        # OPTIMIZER ---------------------------------------------------------------------------------------------------#
        optimizer = optim.Adam(
            craft.parameters(),
            lr=self.config.train.lr,
            weight_decay=self.config.train.weight_decay,
        )

        if self.config.train.ckpt_path is not None and self.config.train.st_iter != 0:
            optimizer.load_state_dict(copyStateDict(self.net_param["optimizer"]))
            self.config.train.st_iter = self.net_param["optimizer"]["state"][0]["step"]
            self.config.train.lr = self.net_param["optimizer"]["param_groups"][0]["lr"]

        # LOSS --------------------------------------------------------------------------------------------------------#
        # Mixed precision
        if self.config.train.amp and torch.cuda.is_available():
            scaler = torch.amp.GradScaler(str(self.device))

            if (
                self.config.train.ckpt_path is not None
                and self.config.train.st_iter != 0
            ):
                scaler.load_state_dict(copyStateDict(self.net_param["scaler"]))
        else:
            scaler = None

        criterion = self.get_loss()

        # TRAIN -------------------------------------------------------------------------------------------------------#
        train_step = self.config.train.st_iter
        whole_training_step = self.config.train.end_iter
        update_lr_rate_step = 0
        training_lr = self.config.train.lr
        loss_value = 0
        batch_time = 0
        start_time = time.time()
        iou_scores = []
        train_loss = []
        train_iou = []
        batch_size = self.config.train.batch_size

        print(
            "================================ Train start ================================"
        )
        while train_step < whole_training_step:
            print("Epoch: {}/{}".format(train_step, whole_training_step))
            for index, (
                images,
                region_scores,
                affinity_scores,
                confidence_masks,
            ) in tqdm(enumerate(trn_real_loader), total=len(trn_real_loader)):
                craft.train()
                if train_step > 0 and train_step % self.config.train.lr_decay == 0:
                    update_lr_rate_step += 1
                    training_lr = self.adjust_learning_rate(
                        optimizer,
                        self.config.train.gamma,
                        update_lr_rate_step,
                        self.config.train.lr,
                    )

                images = images.to(device, non_blocking=torch.cuda.is_available())
                region_scores = region_scores.to(
                    device, non_blocking=torch.cuda.is_available()
                )
                affinity_scores = affinity_scores.to(
                    device, non_blocking=torch.cuda.is_available()
                )
                confidence_masks = confidence_masks.to(
                    device, non_blocking=torch.cuda.is_available()
                )

                if self.config.train.use_synthtext:
                    # Synth image load
                    syn_image, syn_region_label, syn_affi_label, syn_confidence_mask = (
                        next(batch_syn)
                    )
                    syn_image = syn_image.to(
                        device, non_blocking=torch.cuda.is_available()
                    )
                    syn_region_label = syn_region_label.to(
                        device, non_blocking=torch.cuda.is_available()
                    )
                    syn_affi_label = syn_affi_label.to(
                        device, non_blocking=torch.cuda.is_available()
                    )
                    syn_confidence_mask = syn_confidence_mask.to(
                        device, non_blocking=torch.cuda.is_available()
                    )

                    # Concat syn & custom image
                    images = torch.cat((syn_image, images), 0)
                    region_image_label = torch.cat((syn_region_label, region_scores), 0)
                    affinity_image_label = torch.cat(
                        (syn_affi_label, affinity_scores), 0
                    )
                    confidence_mask_label = torch.cat(
                        (syn_confidence_mask, confidence_masks), 0
                    )
                else:
                    region_image_label = region_scores
                    affinity_image_label = affinity_scores
                    confidence_mask_label = confidence_masks

                if self.config.train.amp and torch.cuda.is_available():
                    with torch.amp.autocast(str(self.device)):
                        output, _ = craft(images)
                        out1 = output[:, :, :, 0]
                        out2 = output[:, :, :, 1]

                        loss = criterion(
                            region_image_label,
                            affinity_image_label,
                            out1,
                            out2,
                            confidence_mask_label,
                            self.config.train.neg_rto,
                            self.config.train.n_min_neg,
                        )

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                else:
                    output, _ = craft(images)
                    out1 = output[:, :, :, 0]
                    out2 = output[:, :, :, 1]
                    loss = criterion(
                        region_image_label,
                        affinity_image_label,
                        out1,
                        out2,
                        confidence_mask_label,
                        self.config.train.neg_rto,
                        self.config.train.n_min_neg,
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                iou_region = self.iou_train(out1, region_image_label)
                iou_affinity = self.iou_train(out2, affinity_image_label)
                iou_scores.append((iou_region + iou_affinity) / 2)
                end_time = time.time()
                loss_value += loss.item()
                batch_time += end_time - start_time

                if train_step > 0 and train_step % batch_size == 0:
                    mean_loss = loss_value / batch_size
                    loss_value = 0
                    avg_batch_time = batch_time / batch_size
                    batch_time = 0
                    mean_iou = sum(iou_scores) / len(iou_scores)
                    iou_scores = []
                    train_loss.append(mean_loss)
                    train_iou.append(mean_iou)

                    print(
                        "{}, training_step: {}|{}, learning rate: {:.6f}, "
                        "training_loss: {:.5f}, mean_iou: {:.4f}, avg_batch_time: {:.5f}".format(
                            time.strftime(
                                "%Y-%m-%d:%H:%M:%S", time.localtime(time.time())
                            ),
                            train_step,
                            whole_training_step,
                            training_lr,
                            mean_loss,
                            mean_iou,
                            avg_batch_time,
                        )
                    )

                    if self.config.wandb_opt:
                        wandb.log({"train_step": train_step, "mean_loss": mean_loss})

                if (
                    train_step % self.config.train.eval_interval == 0
                    and train_step != 0
                ):
                    craft.eval()

                    print("Saving state, index:", train_step)
                    save_param_dic = {
                        "iter": train_step,
                        "craft": craft.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    save_param_path = (
                        self.config.results_dir
                        + "/CRAFT_clr_"
                        + repr(train_step)
                        + ".pth"
                    )

                    if self.config.train.amp:
                        save_param_dic["scaler"] = scaler.state_dict()
                        save_param_path = (
                            self.config.results_dir
                            + "/CRAFT_clr_amp_"
                            + repr(train_step)
                            + ".pth"
                        )

                    torch.save(save_param_dic, save_param_path)

                    # validation
                    self.iou_eval(
                        "custom_data",
                        train_step,
                        buffer_dict["custom_data"],
                        craft,
                    )

                train_step += 1
                if train_step >= whole_training_step:
                    break

            if self.config.mode == "weak_supervision":
                state_dict = craft.module.state_dict()
                supervision_model.load_state_dict(state_dict)
                trn_real_dataset.update_model(supervision_model)

        print(
            "================================ Train end ================================"
        )
        self.plot_learning_curves(train_loss, train_iou)
        # save last model
        save_param_dic = {
            "iter": train_step,
            "craft": craft.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_param_path = (
            self.config.results_dir + "/CRAFT_clr_" + repr(train_step) + ".pth"
        )

        if self.config.train.amp:
            save_param_dic["scaler"] = scaler.state_dict()
            save_param_path = (
                self.config.results_dir + "/CRAFT_clr_amp_" + repr(train_step) + ".pth"
            )
        torch.save(save_param_dic, save_param_path)


def main():
    parser = argparse.ArgumentParser(description="CRAFT custom data train")
    parser.add_argument(
        "--yaml",
        "--yaml_file_name",
        default="custom_data_train",
        type=str,
        help="Load configuration",
    )
    parser.add_argument(
        "--port", "--use ddp port", default="2346", type=str, help="Port number"
    )

    args = parser.parse_args()

    # load configure
    exp_name = args.yaml
    config = load_yaml(args.yaml)

    # Make result_dir
    res_dir = os.path.join(config["results_dir"], args.yaml)
    config["results_dir"] = res_dir
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # Duplicate yaml file to result_dir
    shutil.copy(
        "config/" + args.yaml + ".yaml", os.path.join(res_dir, args.yaml) + ".yaml"
    )

    if config["mode"] == "weak_supervision":
        mode = "weak_supervision"
    else:
        mode = None

    # Apply config to wandb
    if config["wandb_opt"]:
        wandb.init(project="craft-stage2", entity="user_name", name=exp_name)
        wandb.config.update(config)

    config = DotDict(config)

    # Start train
    buffer_dict = {"custom_data": None}
    trainer = Trainer(config, 0, mode)
    trainer.train(buffer_dict)

    if config["wandb_opt"]:
        wandb.finish()


if __name__ == "__main__":
    main()
