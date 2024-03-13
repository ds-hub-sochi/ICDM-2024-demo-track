import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from functools import partial
from pathlib import Path
from collections import OrderedDict


from timm.models import create_model

from datasets import build_dataset

# from utils import custom_collate, aug_collate, multi_sample_collate

import utils
import modeling_finetune


def get_args():
    parser = argparse.ArgumentParser(
        'VideoMAE fine-tuning and evaluation script for video classification',
        add_help=False,
    )
    parser.add_argument('--batch_size', default=16, type=int)

    # Model parameters
    parser.add_argument(
        '--model',
        default='vit_large_patch16_224',
        type=str,
        metavar='MODEL',
        help='Name of model to train',
    )
    parser.add_argument('--tubelet_size', type=int, default=2)
    parser.add_argument('--input_size', default=224, type=int, help='videos input size')

    parser.add_argument('--fc_drop_rate', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    parser.add_argument(
        '--attn_drop_rate',
        type=float,
        default=0.0,
        metavar='PCT',
        help='Attention dropout rate (default: 0.)',
    )
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Finetuning params
    parser.add_argument(
        '--finetune',
        default='/home/jovyan/people/milevich/rsl/slt_mae/data/pWLASLfWLASLfSLOVO.pt',
        help='finetune from checkpoint',
    )
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.set_defaults(use_checkpoint=False)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')

    # Dataset parameters
    parser.add_argument('--data_path', default='/path/to/folder_with_video', type=str, help='dataset path')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument(
        '--pin_mem',
        action='store_true',
        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.',
    )
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--short_side_size', type=int, default=224)
    parser.add_argument(
        '--anno_path',
        required=True,
        help='path to annotation in mae format',
    )
    parser.add_argument(
        '--anno_type',
        default='names',
        choices=['names', 'paths'],
        help='wheather annotations contains video names or video paths',
    )
    return parser.parse_args()


def main(args):
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(args=args)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=None,
    )
    return data_loader_train, dataset_train


from tqdm import tqdm


def get_embeds(dataloader):
    embeds = torch.tensor([]).to('cuda:0')
    labels = torch.tensor([])
    model.to('cuda:0')
    model.eval()
    with torch.inference_mode():
        for i, data in tqdm(enumerate(dataloader)):
            batch_embeds = model.forward_features(data[0].to('cuda:0'))
            embeds = torch.cat((embeds, batch_embeds), 0)
            labels = torch.cat((labels, data[1]))
            # if i>2:
            #     break
    return embeds.cpu().numpy(), labels.cpu().numpy().astype(int)


if __name__ == '__main__':
    opts = get_args()
    train_loader, dataset_train = main(opts)
    args = opts

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        all_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        fc_drop_rate=args.fc_drop_rate,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_checkpoint=args.use_checkpoint,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
    )

    patch_size = model.patch_embed.patch_size
    print('Patch size = %s' % str(patch_size))
    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print('Load ckpt from %s' % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print('Load state_dict by model_key = %s' % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f'Removing key {k} from pretrained checkpoint')
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict

        # interpolate position embedding
        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]  # channel dim
            num_patches = model.patch_embed.num_patches  #
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches  # 0/1

            # height (== width) for the checkpoint position embedding
            orig_size = int(
                (
                    (pos_embed_checkpoint.shape[-2] - num_extra_tokens)
                    // (args.num_frames // model.patch_embed.tubelet_size)
                )
                ** 0.5,
            )
            # height (== width) for the new position embedding
            new_size = int((num_patches // (args.num_frames // model.patch_embed.tubelet_size)) ** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print('Position interpolate from %dx%d to %dx%d' % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                # B, L, C -> BT, H, W, C -> BT, C, H, W
                pos_tokens = pos_tokens.reshape(
                    -1,
                    args.num_frames // model.patch_embed.tubelet_size,
                    orig_size,
                    orig_size,
                    embedding_size,
                )
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens,
                    size=(new_size, new_size),
                    mode='bicubic',
                    align_corners=False,
                )
                # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(
                    -1,
                    args.num_frames // model.patch_embed.tubelet_size,
                    new_size,
                    new_size,
                    embedding_size,
                )
                pos_tokens = pos_tokens.flatten(1, 3)  # B, L, C
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed

        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    embeds, labels = get_embeds(train_loader)
    save_path = Path(args.anno_path).parent.absolute()
    np.save(os.path.join(save_path, 'embeds.npy'), embeds)
    np.save(os.path.join(save_path, 'labels.npy'), labels)
