import os
import random
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import commons
import utils
from losses import discriminator_loss, feature_loss, generator_loss
from meldataset import MelDataset, custom_data_load, mel_spectrogram
from models_bigvgan import Generator, MultiPeriodDiscriminator

torch.backends.cudnn.benchmark = True
global_step = 0


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()

    print("Using Num. GPUs:", n_gpus)

    # port = 50000 + random.randint(0, 100)
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = str(port)

    hps = utils.get_hparams()

    hps.train.batch_size = int(hps.train.batch_size / n_gpus)
    print("Batch size per GPU :", hps.train.batch_size)

    mp.spawn(
        run,
        nprocs=n_gpus,
        args=(
            n_gpus,
            hps,
        ),
    )


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(
        backend="nccl",
        init_method="tcp://localhost:54321",
        world_size=n_gpus,
        rank=rank,
    )
    print("Process Group Created: ", rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)
    device = torch.device("cuda:{:d}".format(rank))

    net_g = Generator(
        hps.data.n_mel_channels,
        resblock_kernel_sizes=hps.model.resblock_kernel_sizes,
        resblock_dilation_sizes=hps.model.resblock_dilation_sizes,
        upsample_rates=hps.model.upsample_rates,
        upsample_initial_channel=hps.model.upsample_initial_channel,
        upsample_kernel_sizes=hps.model.upsample_kernel_sizes,
        gin_channels=0,
        rank=rank,
    ).cuda(rank)

    if rank == 0:
        num_param = get_param_num(net_g)
        print("Number of Parameters:", num_param)

    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    try:
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g
        )
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d
        )
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        epoch_str = -1
        global_step = 0

    validation_filelist, training_filelist = custom_data_load(5)

    print("Training Filelist:", len(training_filelist))
    print("Validation Filelist:", len(validation_filelist))

    trainset = MelDataset(
        training_filelist,
        hps.train.segment_size,
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.sampling_rate,
        hps.data.mel_fmin,
        hps.data.mel_fmax,
        n_cache_reuse=0,
        shuffle=False if n_gpus > 1 else True,
        fmax_loss=hps.data.fmax_for_loss,
        device=device,
    )

    train_sampler = DistributedSampler(trainset) if n_gpus > 1 else None

    train_loader = DataLoader(
        trainset,
        num_workers=8,
        shuffle=False,
        sampler=train_sampler,
        batch_size=hps.train.batch_size,
        pin_memory=True,
    )

    if rank == 0:
        validset = MelDataset(
            validation_filelist,
            hps.train.segment_size,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.sampling_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
            False,
            False,
            n_cache_reuse=0,
            fmax_loss=hps.data.fmax_for_loss,
            device=device,
        )
        eval_loader = DataLoader(
            validset,
            num_workers=1,
            shuffle=False,
            sampler=None,
            batch_size=1,
            pin_memory=True,
            drop_last=True,
        )

    if n_gpus > 1:
        print("Using Distributed Data Loader")
        net_g = DDP(net_g, device_ids=[rank]).to(rank)
        net_d = DDP(net_d, device_ids=[rank]).to(rank)

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)

    net_g.train()
    net_d.train()

    for epoch in range(max(0, epoch_str), hps.train.epochs):

        if n_gpus > 1:
            train_sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            x, y, _, y_mel = batch
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            y = y.unsqueeze(1)

            with autocast(enabled=hps.train.fp16_run):

                y_hat = net_g(x)

                with autocast(enabled=False):
                    y_g_hat_mel = mel_spectrogram(
                        y_hat.float().squeeze(1),
                        hps.data.filter_length,
                        hps.data.n_mel_channels,
                        hps.data.sampling_rate,
                        hps.data.hop_length,
                        hps.data.win_length,
                        hps.data.mel_fmin,
                        hps.data.mel_fmax,
                    )

                # Discriminator
                y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
                with autocast(enabled=False):
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                        y_d_hat_r, y_d_hat_g
                    )
                    loss_disc_all = loss_disc

            optim_d.zero_grad()
            scaler.scale(loss_disc_all).backward()
            scaler.unscale_(optim_d)
            scaler.step(optim_d)

            with autocast(enabled=hps.train.fp16_run):
                # Generator
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
                with autocast(enabled=False):

                    loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * hps.train.c_mel
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)
                    loss_gen_all = loss_gen + loss_fm + loss_mel

            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            scaler.step(optim_g)
            scaler.update()

            if rank == 0:
                if global_step % hps.train.log_interval == 0:
                    lr = optim_g.param_groups[0]["lr"]
                    # losses = [loss_disc, loss_gen, loss_fm, loss_mel]
                    # logger.info(
                    #     "Train Epoch: {} [{:.0f}%]".format(
                    #         epoch, 100.0 * batch_idx / len(train_loader)
                    #     )
                    # )
                    print(
                        "Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}, lr : {:4.5f}".format(
                            global_step,
                            loss_gen_all,
                            loss_mel,
                            time.time() - start_b,
                            lr,
                        )
                    )
                    # logger.info(
                    #     [x.item() for x in losses]
                    #     + [global_step, lr, time.time() - start_b]
                    # )

                    scalar_dict = {
                        "loss/g/total": loss_gen_all,
                        "loss/d/total": loss_disc_all,
                        "learning_rate": lr,
                    }
                    scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel})

                    scalar_dict.update(
                        {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                    )
                    scalar_dict.update(
                        {
                            "loss/d_r/{}".format(i): v
                            for i, v in enumerate(losses_disc_r)
                        }
                    )
                    scalar_dict.update(
                        {
                            "loss/d_g/{}".format(i): v
                            for i, v in enumerate(losses_disc_g)
                        }
                    )
                    image_dict = {
                        "slice/mel_org": utils.plot_spectrogram_to_numpy(
                            y_mel[0].data.cpu().numpy()
                        ),
                        "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                            y_g_hat_mel[0].data.cpu().numpy()
                        ),
                        # "all/mel": utils.plot_spectrogram_to_numpy(
                        #     mel[0].data.cpu().numpy()
                        # ),
                    }
                    utils.summarize(
                        writer=writer,
                        global_step=global_step,
                        images=image_dict,
                        scalars=scalar_dict,
                    )

                if global_step % hps.train.checkpoint_interval == 0:
                    utils.save_checkpoint(
                        net_g,
                        optim_g,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
                    )
                    utils.save_checkpoint(
                        net_d,
                        optim_d,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
                    )

            global_step += 1

        if rank == 0:
            net_g.eval()
            torch.cuda.empty_cache()
            with torch.no_grad():
                for batch_idx, batch in enumerate(eval_loader):
                    x, y, _, _ = batch

                    y_hat = net_g(x.to(rank))

                    y_g_hat_mel = mel_spectrogram(
                        y_hat.squeeze(1),
                        hps.data.filter_length,
                        hps.data.n_mel_channels,
                        hps.data.sampling_rate,
                        hps.data.hop_length,
                        hps.data.win_length,
                        hps.data.mel_fmin,
                        hps.data.mel_fmax,
                    )

                    if batch_idx <= 3:
                        image_dict = {
                            "gen/mel": utils.plot_spectrogram_to_numpy(
                                y_g_hat_mel[0].cpu().numpy()
                            )
                        }

                        audio_dict = {"gen/audio_{}".format(batch_idx): y_hat[0]}
                        if global_step == 0:
                            image_dict.update(
                                {
                                    "gt/mel": utils.plot_spectrogram_to_numpy(
                                        x[0].cpu().numpy()
                                    )
                                }
                            )
                            audio_dict.update({"gt/audio_{}".format(batch_idx): y[0]})

                        utils.summarize(
                            writer=writer_eval,
                            global_step=global_step,
                            images=image_dict,
                            audios=audio_dict,
                            audio_sampling_rate=hps.data.sampling_rate,
                        )
            net_g.train()

        if rank == 0:
            logger.info("====> Epoch: {}".format(epoch))

        scheduler_g.step()
        scheduler_d.step()


if __name__ == "__main__":
    main()
