import os
import shutil
from enum import Enum

import torch

__all__ = ["accuracy", "make_directory", "Summary", "AverageMeter", "ProgressMeter"]


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(correct_k.mul_(100.0 / batch_size))
        return results


def make_directory(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmt_map = {
            Summary.NONE: "",
            Summary.AVERAGE: "{name} {avg:.2f}",
            Summary.SUM: "{name} {sum:.2f}",
            Summary.COUNT: "{name} {count:.2f}"
        }
        if self.summary_type not in fmt_map:
            raise ValueError(f"Invalid summary type {self.summary_type}")
        return fmt_map[self.summary_type].format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self, type_work='train'):
        filepath = f"./points/Adam_{type_work}.csv"
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

        # entries = [f"{meter.avg:.2f}" for meter in self.meters]  # Collect averages only
        # summary = ",".join(entries)  # CSV format: values separated by commas
        #
        # # Write summary to a CSV file
        # with open(filepath, "a") as f:  # 'a' mode to append to the file
        #     f.write(summary + "\n")

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
