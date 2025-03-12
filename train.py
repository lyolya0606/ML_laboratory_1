import os
import time

import torch
from torch import nn, optim
from torch import amp
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from AdamW import AdamWCustom
from dataset import ImageDataset
from utils import accuracy, make_directory, save_checkpoint, AverageMeter, ProgressMeter
import config
import model


def main():
    start_epoch = 0
    best_acc1 = 0.0

    train_dataloader, valid_dataloader = load_dataset()
    googlenet_model, ema_googlenet_model = build_model()
    pixel_criterion = define_loss()
    optimizer = define_optimizer(googlenet_model)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, config.lr_scheduler_T_0, config.lr_scheduler_T_mult, config.lr_scheduler_eta_min)

    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    make_directory(samples_dir)
    make_directory(results_dir)

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))

    # Initialize the gradient scaler
    scaler = amp.GradScaler('cuda')

    for epoch in range(start_epoch, config.epochs):
        train(googlenet_model, ema_googlenet_model, train_dataloader, pixel_criterion, optimizer, epoch, scaler, writer)
        acc1 = validate(ema_googlenet_model, valid_dataloader, epoch, writer, "Valid")
        print("\n")

        scheduler.step()

        is_best = acc1 > best_acc1
        is_last = (epoch + 1) == config.epochs
        best_acc1 = max(acc1, best_acc1)
        save_checkpoint({"epoch": epoch + 1, "best_acc1": best_acc1, "state_dict": googlenet_model.state_dict(),
                         "ema_state_dict": ema_googlenet_model.state_dict(), "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict()}, f"epoch_{epoch + 1}.pth.tar",
                        samples_dir, results_dir, is_best, is_last)


def load_dataset() -> [DataLoader, DataLoader]:
    train_dataset = ImageDataset(config.train_image_dir, config.train_annotation_path, config.image_size, "Train")
    valid_dataset = ImageDataset(config.valid_image_dir, config.valid_annotation_path, config.image_size, "Valid")

    # Generator all dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                  num_workers=config.num_workers, pin_memory=True, drop_last=True, persistent_workers=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False,
                                  num_workers=config.num_workers, pin_memory=True, drop_last=False, persistent_workers=True)

    return train_dataloader, valid_dataloader


def build_model() -> [nn.Module, nn.Module]:
    googlenet_model = model.__dict__[config.model_arch_name](num_classes=config.model_num_classes,
                                                             aux_logits=True,
                                                             transform_input=True)
    googlenet_model = googlenet_model.to(device=config.device, memory_format=torch.channels_last)

    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: (1 - config.model_ema_decay) * averaged_model_parameter + config.model_ema_decay * model_parameter
    ema_googlenet_model = AveragedModel(googlenet_model, avg_fn=ema_avg)

    return googlenet_model, ema_googlenet_model


def define_loss() -> nn.CrossEntropyLoss:
    criterion = nn.CrossEntropyLoss(label_smoothing=config.loss_label_smoothing)
    criterion = criterion.to(device=config.device, memory_format=torch.channels_last)

    return criterion


def define_optimizer(model) -> AdamWCustom:
    optimizer = AdamWCustom(model.parameters(),
                            lr=1e-4,
                            betas=(0.9, 0.999),
                            eps=1e-8,
                            weight_decay=config.model_weight_decay)
    return optimizer

# def define_optimizer(model) -> optim.SGD:
#     optimizer = optim.SGD(model.parameters(),
#                           lr=config.model_lr,
#                           momentum=config.model_momentum,
#                           weight_decay=config.model_weight_decay)
#
#     return optimizer

# def define_optimizer(model) -> optim.Adam:
#     optimizer = optim.Adam(model.parameters(),
#                             lr=1e-4,
#                             weight_decay=config.model_weight_decay,
#                             betas=(0.9, 0.999),
#                             eps=1e-8
#                             )
#     return optimizer


def train(model, ema_model, train_dataloader, criterion, optimizer, epoch, scaler, writer):
    model.train()
    batches = len(train_dataloader)
    progress = ProgressMeter(
        batches,
        [AverageMeter("Time", ":6.3f"), AverageMeter("Data", ":6.3f"),
         AverageMeter("Loss", ":6.6f"), AverageMeter("Acc@1", ":6.2f"),
         AverageMeter("Acc@5", ":6.2f")],
        prefix=f"Epoch: [{epoch + 1}]"
    )

    end = time.time()
    for batch_index, batch_data in enumerate(train_dataloader):
        # Замер времени загрузки данных
        progress.meters[1].update(time.time() - end)

        # Перенос данных на устройство
        images = batch_data["image"].to(config.device, memory_format=torch.channels_last, non_blocking=True)
        target = batch_data["target"].to(config.device, non_blocking=True)

        # Обнуление градиентов
        model.zero_grad(set_to_none=True)

        # Смешанная точность и вычисление потерь
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = sum(w * criterion(o, target) for w, o in zip(
                [config.loss_aux3_weights, config.loss_aux2_weights, config.loss_aux1_weights], output
            ))

        # Обратное распространение и обновление весов
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Обновление EMA модели
        ema_model.update_parameters(model)

        # Вычисление точности
        top1, top5 = accuracy(output[0], target, topk=(1, 5))
        progress.meters[2].update(loss.item(), images.size(0))
        progress.meters[3].update(top1[0].item(), images.size(0))
        progress.meters[4].update(top5[0].item(), images.size(0))

        # Замер времени выполнения батча
        progress.meters[0].update(time.time() - end)
        end = time.time()

        # Логирование
        if batch_index % config.train_print_frequency == 0:
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches)
            progress.display(batch_index + 1)


def validate(ema_model, valid_dataloader, epoch, writer, mode):
    ema_model.eval()
    batches = len(valid_dataloader)
    progress = ProgressMeter(
        batches,
        [AverageMeter("Time", ":6.3f"), AverageMeter("Acc@1", ":6.2f"),
         AverageMeter("Acc@5", ":6.2f")],
        prefix=f"{mode}: "
    )

    end = time.time()
    with torch.no_grad():
        for batch_index, batch_data in enumerate(valid_dataloader):
            # Перенос данных на устройство
            images = batch_data["image"].to(config.device, memory_format=torch.channels_last, non_blocking=True)
            target = batch_data["target"].to(config.device, non_blocking=True)

            # Инференс
            output = ema_model(images)

            # Вычисление точности
            top1, top5 = accuracy(output, target, topk=(1, 5))
            progress.meters[1].update(top1[0].item(), images.size(0))
            progress.meters[2].update(top5[0].item(), images.size(0))

            # Замер времени выполнения батча
            progress.meters[0].update(time.time() - end)
            end = time.time()

            # Логирование
            if batch_index % config.valid_print_frequency == 0:
                progress.display(batch_index + 1)

    # Итоговое отображение метрик
    progress.display_summary()
    if mode in ["Valid", "Test"]:
        writer.add_scalar(f"{mode}/Acc@1", progress.meters[1].avg, epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return progress.meters[1].avg


if __name__ == "__main__":
    main()