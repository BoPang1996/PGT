import os
import argparse
from tensorboardX import SummaryWriter

from slowfast.utils.parser import load_config, parse_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    filename = os.path.join(args.log_dir, 'train.log')
    tblogger = SummaryWriter(log_dir=args.log_dir)

    with open(filename) as f:
        log = f.readlines()

    for l in log:
        if 'train_iter' in l:
            l = l.split(']: ')[1].strip()
            cur_epoch = int(l.split('epoch: ')[1].split('/')[0])
            cur_iter = int(l.split('iter: ')[1].split('/')[0])
            epoch_iters = int(l.split('iter: ')[1].split('/')[1].split(';')[0])
            iters = cur_iter + 1 + epoch_iters * cur_epoch
            for kv in l.split('; '):
                k, v = kv.split(': ')
                if 'err' in k or 'loss' in k:
                    tblogger.add_scalar('train/{}'.format(k), float(v), iters)
                elif k == 'epoch':
                    tblogger.add_scalar('other/epoch', cur_epoch + 1, iters)
                elif k == 'lr':
                    tblogger.add_scalar('other/lr', float(v), iters)
                else:
                    continue
        elif 'val_epoch' in l:
            l = l.split(']: ')[1].strip()
            cur_epoch = int(l.split('epoch: ')[1].split('/')[0])
            for kv in l.split('; '):
                k, v = kv.split(': ')
                if 'err' in k or 'map' in k:
                    tblogger.add_scalar(
                        'val/{}'.format(k), float(v), cur_epoch + 1)
                else:
                    continue


if __name__ == '__main__':
    main()
