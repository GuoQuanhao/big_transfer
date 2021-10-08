# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Fine-tune a BiT model on some downstream dataset."""
#!/usr/bin/env python3
# coding: utf-8
from os.path import join as pjoin  # pylint: disable=g-importing-member
import os
import time

import numpy as np
import paddle
from paddle.io import DataLoader, RandomSampler
import paddle.vision as pv

# import bit_paddle.fewshot as fs
import bit_paddle.lbtoolbox as lb
import bit_paddle.models as models

import bit_common
import bit_hyperrule


def topk(output, target, ks=(1,)):
    """Returns one boolean vector for each k, whether the target is within the output's top-k."""
    _, pred = output.topk(max(ks), 1, True, True)
    pred = pred.t()
    correct = pred.equal(target.reshape([1, -1]).expand_as(pred))
    return [np.max(correct.numpy()[:k], 0) for k in ks]


def recycle(iterable):
    """Variant of itertools.cycle that does not save iterates."""
    while True:
        for i in iterable:
            yield i


def mktrainval(args, logger):
    """Returns train and validation datasets."""
    precrop, crop = bit_hyperrule.get_resolution_from_dataset(args.dataset)
    train_tx = pv.transforms.Compose([
        pv.transforms.Resize((precrop, precrop)),
        pv.transforms.RandomCrop((crop, crop)),
        pv.transforms.RandomHorizontalFlip(),
        pv.transforms.Transpose(),
        pv.transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5)),
    ])
    val_tx = pv.transforms.Compose([
        pv.transforms.Resize((crop, crop)),
        pv.transforms.Transpose(),
        pv.transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5)),
    ])

    if args.dataset == "cifar10":
        classes = 10
        train_set = pv.datasets.Cifar10(transform=train_tx, mode='train', download=True)
        valid_set = pv.datasets.Cifar10(transform=val_tx, mode='test', download=True)
    elif args.dataset == "cifar100":
        classes = 100
        train_set = pv.datasets.Cifar100(transform=train_tx, mode='train', download=True)
        valid_set = pv.datasets.Cifar100(transform=val_tx, mode='test', download=True)
    elif args.dataset == "imagenet2012":
        classes = 1
        train_set = pv.datasets.DatasetFolder(pjoin(args.datadir, "train"), train_tx)
        valid_set = pv.datasets.DatasetFolder(pjoin(args.datadir, "val"), val_tx)
    else:
        raise ValueError(f"Sorry, we have not spent time implementing the "
                         f"{args.dataset} dataset in the Paddle codebase. "
                         f"In principle, it should be easy to add :)")

    logger.info(f"Using a training set with {len(train_set)} images.")
    logger.info(f"Using a validation set with {len(valid_set)} images.")

    micro_batch_size = args.batch // args.batch_split

    valid_loader = DataLoader(
        valid_set, batch_size=micro_batch_size, shuffle=False,
        num_workers=args.workers, drop_last=False)

    if micro_batch_size <= len(train_set):
        train_loader = DataLoader(
            train_set, batch_size=micro_batch_size, shuffle=True,
            num_workers=args.workers, drop_last=False)
    else:
        # In the few-shot cases, the total dataset size might be smaller than the batch-size.
        # In these cases, the default sampler doesn't repeat, so we need to make it do that
        # if we want to match the behaviour from the paper.
        train_loader = DataLoader(
            train_set, batch_size=micro_batch_size, num_workers=args.workers,
            sampler=RandomSampler(train_set, replacement=True, num_samples=micro_batch_size))

    return train_set, valid_set, train_loader, valid_loader, classes


def run_eval(model, data_loader, chrono, logger, step):
    # switch to evaluate mode
    model.eval()

    logger.info("Running validation...")
    logger.flush()

    all_c, all_top1, all_top5 = [], [], []
    end = time.time()
    for b, (x, y) in enumerate(data_loader):
        with paddle.no_grad():
            # measure data loading time
            chrono._done("eval load", time.time() - end)

            # compute output, measure accuracy and record loss.
            with chrono.measure("eval fprop"):
                logits = model(x)
                c = paddle.nn.CrossEntropyLoss(reduction='none')(logits, y)
                top1, top5 = topk(logits, y, ks=(1, 5))
                all_c.extend(c.cpu().numpy())  # Also ensures a sync point.
                all_top1.extend(top1)
                all_top5.extend(top5)

        # measure elapsed time
        end = time.time()

    model.train()
    logger.info(f"Validation@{step} loss {np.mean(all_c):.5f}, "
                f"top1 {np.mean(all_top1):.2%}, "
                f"top5 {np.mean(all_top5):.2%}")
    logger.flush()


def mixup_data(x, y, l):
    """Returns mixed inputs, pairs of targets, and lambda"""
    indices = np.random.permutation(x.shape[0])

    mixed_x = l * x + (1 - l) * paddle.to_tensor(x.numpy()[indices])
    y_a, y_b = y, paddle.to_tensor(y.numpy()[indices])
    return mixed_x, y_a, y_b


def mixup_criterion(criterion, pred, y_a, y_b, l):
    return l * criterion(pred, y_a) + (1 - l) * criterion(pred, y_b)


def main(args):
    logger = bit_common.setup_logger(args)

    # Only good if sizes stay the same within the main loop!
    train_set, valid_set, train_loader, valid_loader, classes = mktrainval(args, logger)

    supports = bit_hyperrule.get_schedule(len(train_set))

    logger.info(f"Loading model from {args.model}.npz")
    model = models.KNOWN_MODELS[args.model](head_size=classes, zero_head=True)
    model.load_from(np.load(f"{args.model}.npz"))

    logger.info("Moving model onto all GPUs")
    # model = paddle.nn.DataParallel(model)

    # Optionally resume from a checkpoint.
    # Load it to CPU first as we'll move the model to GPU later.
    # This way, we save a little bit of GPU memory when loading.
    # Note: no weight-decay!
    step = 0
    scheduler = paddle.optimizer.lr.StepDecay(learning_rate=args.base_lr, step_size=supports[2]-supports[1], gamma=0.1, last_epoch=- 1, verbose=False)
    warm_up = paddle.optimizer.lr.LinearWarmup(learning_rate=scheduler, warmup_steps=supports[0], start_lr=0, end_lr=args.base_lr, verbose=False)
    optim = paddle.optimizer.Momentum(parameters=model.parameters(), learning_rate=warm_up)

    # Resume fine-tuning if we find a saved model.
    savename = pjoin(args.logdir, args.name, "bit.pdparams")
    optname = pjoin(args.logdir, args.name, "bit.pdopt")


    logger.info(f"Model will be saved in '{savename}'")
    try:
        checkpoint = paddle.load(savename)
        opt_checkpoint = paddle.load(optname)
        logger.info(f"Found saved model to resume from at '{savename}'")

        step = opt_checkpoint['LR_Scheduler']['last_epoch']
        model.set_state_dict(checkpoint)
        optim.set_state_dict(opt_checkpoint)
        logger.info(f"Resumed at step {step}")
    except ValueError:
        logger.info("Fine-tuning from BiT")

    optim.clear_grad()

    model.train()
    mixup = bit_hyperrule.get_mixup(len(train_set))
    cri = paddle.nn.CrossEntropyLoss()

    logger.info("Starting training!")
    chrono = lb.Chrono()
    accum_steps = 0
    mixup_l = np.random.beta(mixup, mixup) if mixup > 0 else 1
    end = time.time()

    with lb.Uninterrupt() as u:
        for x, y in recycle(train_loader):
            # measure data loading time, which is spent in the `for` statement.
            chrono._done("load", time.time() - end)

            if u.interrupted:
                break

            if mixup > 0.0:
                x, y_a, y_b = mixup_data(x, y, mixup_l)

            # compute output
            with chrono.measure("fprop"):

                logits = model(x)
                if mixup > 0.0:
                    c = mixup_criterion(cri, logits, y_a, y_b, mixup_l)
                else:
                    c = cri(logits, y)
                c_num = float(c.cpu().numpy())  # Also ensures a sync point.

            # Accumulate grads
            with chrono.measure("grads"):
                (c / args.batch_split).backward()
                accum_steps += 1

            accstep = f" ({accum_steps}/{args.batch_split})" if args.batch_split > 1 else ""
            logger.info(f"[step {step}{accstep}]: loss={c_num:.5f} (lr={optim._learning_rate():.1e})")  # pylint: disable=logging-format-interpolation
            logger.flush()

            # Update params
            if accum_steps == args.batch_split:
                with chrono.measure("update"):
                    optim.step()
                    optim.clear_grad()
                    warm_up.step()
                step += 1
                accum_steps = 0
                # Sample new mixup ratio for next batch
                mixup_l = np.random.beta(mixup, mixup) if mixup > 0 else 1

                # Run evaluation and save the model.
                if args.eval_every and step % args.eval_every == 0:
                    run_eval(model, valid_loader, chrono, logger, step)
                    if args.save:
                        paddle.save(model.state_dict(), savename)
                        paddle.save(optim.state_dict(), optname)
            end = time.time()
            if step == supports[-1]:
                break
                # Final eval at end of training.
        run_eval(model, valid_loader, chrono, logger, step='end')

    logger.info(f"Timings:\n{chrono}")


if __name__ == "__main__":
    parser = bit_common.argparser(models.KNOWN_MODELS.keys())
    parser.add_argument("--datadir", required=True,
                      help="Path to the ImageNet data folder, preprocessed for paddlevision.")
    parser.add_argument("--workers", type=int, default=0,
                      help="Number of background threads used to load data.")
    parser.add_argument("--no-save", dest="save", action="store_false")
    main(parser.parse_args())
