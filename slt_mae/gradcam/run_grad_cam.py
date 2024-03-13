import argparse
import torch
import torch.backends.cudnn as cudnn
import utils
import modeling_finetune
from PIL import Image
from pathlib import Path
from collections import OrderedDict
from timm.models import create_model
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from decord import VideoReader, cpu
from grad_cam import GradCAM
from torchvision import transforms
from transforms import GroupNormalize, Stack, ToTorchFormatTensor, GroupCenterCrop


def get_args():
    parser = argparse.ArgumentParser('VideoMAE fine-tuning and evaluation script for video classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=100, type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--tubelet_size', type=int, default= 2)
    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size')

    parser.add_argument('--fc_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--finetune', default='/home/jovyan/murtazin/slt_mae/workdirs/videomae_pretrain_large_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1659_pWLASL2fSLOVO/checkpoint-best/mp_rank_00_model_states.pt', help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.set_defaults(use_checkpoint=False)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument('--sampling_type', default='circle', choices=['circle', 'last'], help='.....')

    # Dataset parameters
    parser.add_argument('--img_path', default='/home/jovyan/murtazin/slt_mae/data/1.mp4', type=str,
                        help='dataset path')
    parser.add_argument('--save_path', default='/home/jovyan/murtazin/slt_mae/result_video', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1001, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--num_segments', type=int, default=1)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    return parser.parse_args()
    
def get_model(args):
    print(args)
    cudnn.benchmark = True
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        all_frames=args.num_frames * args.num_segments,
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
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
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
            embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
            num_patches = model.patch_embed.num_patches # 
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 0/1

            # height (== width) for the checkpoint position embedding 
            orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(args.num_frames // model.patch_embed.tubelet_size)) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int((num_patches // (args.num_frames // model.patch_embed.tubelet_size) )** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                # B, L, C -> BT, H, W, C -> BT, C, H, W
                pos_tokens = pos_tokens.reshape(-1, args.num_frames // model.patch_embed.tubelet_size, orig_size, orig_size, embedding_size)
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, args.num_frames // model.patch_embed.tubelet_size, new_size, new_size, embedding_size) 
                pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed

        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
    return model

class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406] # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225] # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupCenterCrop(args.input_size)
        self.transform = transforms.Compose([
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])

    def __call__(self, images):
        process_data, _ = self.transform(images)
        return process_data

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += ")"
        return repr


def main(args):
    print(args)
    device = torch.device(args.device)
    cudnn.benchmark = True
    model = get_model(args)
    model.to(device)
    model.eval
    print(model)

    
    if args.save_path:
        Path(args.save_path).mkdir(parents=True, exist_ok=True)

    restored_video = []
    masked_video = []
    orig_video = []
 
    with open(args.img_path, 'rb') as f:
        vr = VideoReader(f, ctx=cpu(0))
    batches = [list(range(i*args.num_frames, (i*args.num_frames)+args.num_frames)) for i in range(len(vr) // args.num_frames)]

    data_mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
    data_std=torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]
    grad_cam = GradCAM(model=model, target_layers=['norm'], data_mean=data_mean, data_std=data_std)

    # normalize = GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # transforms = transforms.Compose([
    #             Stack(roll=False),
    #             ToTorchFormatTensor(div=True),
    #             normalize,
    #             ])
    
    for batch_id, frame_id_list in enumerate(batches):
        print(len(batches))
        video_data = vr.get_batch(frame_id_list).asnumpy()
        img = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid, _ in enumerate(frame_id_list)]
        img = [i.resize((224,224)) for i in img]
        transforms = DataAugmentationForVideoMAE(args)
        img = transforms((img, None)) # T*C,H,W
        img = img.view((args.num_frames , 3) + img.size()[-2:]).transpose(0,1) # T*C,H,W -> T,C,H,W -> C,T,H,W
        # img = img.view(( -1 , args.num_frames) + img.size()[-2:])

        img = img.unsqueeze(0)
        results = grad_cam(img)

        # print(results)
        # #save original video
        # mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
        # std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]
        # ori_img = img * std + mean  # in [0, 1]
        # imgs = [ToPILImage()(ori_img[0,:,vid,:,:].cpu()) for vid, _ in enumerate(frame_id_list)]
        # # for id, im in enumerate(imgs):
        # #     im.save(f"{args.save_path}/ori_img{id}_{batch_id}.jpg")

        # img_squeeze = rearrange(ori_img, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size[0], p2=patch_size[0])
        # img_norm = (img_squeeze - img_squeeze.mean(dim=-2, keepdim=True)) / (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
        # img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')
        # img_patch[bool_masked_pos] = outputs

        # #make mask
        # mask = torch.ones_like(img_patch)
        # mask[bool_masked_pos] = 0
        # mask = rearrange(mask, 'b n (p c) -> b n p c', c=3)
        # mask = rearrange(mask, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2) ', p0=2, p1=patch_size[0], p2=patch_size[1], h=14, w=14)

        # #save reconstruction video
        # rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=3)
        # # Notice: To visualize the reconstruction video, we add the predict and the original mean and var of each patch.
        # rec_img = rec_img * (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + img_squeeze.mean(dim=-2, keepdim=True)
        # rec_img = rearrange(rec_img, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)', p0=2, p1=patch_size[0], p2=patch_size[1], h=14, w=14)
        # imgs = [ ToPILImage()(rec_img[0, :, vid, :, :].cpu().clamp(0,0.996)) for vid, _ in enumerate(frame_id_list)  ]
        # restored_video += imgs
        # # for id, im in enumerate(imgs):
        # #     im.save(f"{args.save_path}/rec_img{id}_{batch_id}.jpg")

        # #save masked video
        # img_mask = rec_img * mask
        # imgs = [ToPILImage()(img_mask[0, :, vid, :, :].cpu()) for vid, _ in enumerate(frame_id_list)]
        # masked_video += imgs

        # # for id, im in enumerate(imgs):
        # #     im.save(f"{args.save_path}/mask_img{id}_{batch_id}.jpg")


    # output_video = args.save_path
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # frame_width, frame_height = restored_video[0].size[0] + masked_video[0].size[0], max(restored_video[0].size[1], masked_video[0].size[1])
    # print(output_video)
    # # Initialize the video writer
    # out = cv2.VideoWriter("result_video/output_75_19epoch.mov", fourcc, 20, (frame_width, frame_height))

    # try:
    #     # Iterate through the images in both lists and combine them side by side
    #     for img1, img2 in zip(restored_video, masked_video):
    #         # Convert PIL images to numpy arrays
    #         frame = np.concatenate((np.array(img1), np.array(img2)), axis=1)

    #         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #         out.write(frame)
    #     out.release()
    #     print("Video saved successfully.")
    # except Exception as e:
    #     print(f"Error: {str(e)}")



if __name__ == '__main__':
    opts = get_args()
    main(opts)
