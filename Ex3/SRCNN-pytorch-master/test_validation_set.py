import argparse

import torch
import torch.backends.cudnn as cudnn

from torch.utils.data.dataloader import DataLoader

from models import SRCNN
from datasets import EvalDataset
from utils import AverageMeter, calc_psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = SRCNN().to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    model.eval()
    epoch_psnr = AverageMeter()

    for data in eval_dataloader:
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(inputs).clamp(0.0, 1.0)

        epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

    print('avg psnr: {:.2f}'.format(epoch_psnr.avg))
