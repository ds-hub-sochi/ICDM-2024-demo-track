import argparse
import torch
import torch.backends.cudnn as cudnn
import utils
import modeling_finetune
import numpy as np
from PIL import Image
from pathlib import Path
from collections import OrderedDict
from timm.models import create_model
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from decord import VideoReader, cpu
from grad_cam import GradCAM
from torchvision import transforms
from transforms import GroupNormalize, Stack, ToTorchFormatTensor, GroupCenterCrop
from pytorch_grad_cam import GradCAM

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

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.feature_maps = []
        self.gradients = []

        # hook to save the output of the last convolution layer
        self.model.patch_embed.proj.register_forward_hook(self.save_feature_maps)
        self.model.patch_embed.proj.register_backward_hook(self.save_gradients)

    def save_feature_maps(self, module, input, output):
        self.feature_maps.append(output)

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0])

    def forward(self, x):
        self.model.eval()
        output = self.model(x)
        return output
      
    def compute_cam(self, target_class):
        one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        self.model.zero_grad()
        output.backward(gradient=one_hot_output.to(device), retain_graph=True)

        gradients = self.gradients[-1].cpu().data.numpy()
        feature_maps = self.feature_maps[-1].cpu().data.numpy()

        weights = np.mean(gradients, axis=(2, 3, 4))  # global average pooling
        cam = np.sum(np.multiply(weights, feature_maps), axis=1)

        # Relu
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)

        return cam
    
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

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.feature_maps = []
        self.gradients = []

        # hook to save the output of the last convolution layer
        self.model.patch_embed.proj.register_forward_hook(self.save_feature_maps)
        self.model.patch_embed.proj.register_backward_hook(self.save_gradients)

    def save_feature_maps(self, module, input, output):
        self.feature_maps.append(output)

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0])

    def forward(self, x):
        self.model.eval()
        output = self.model(x)
        return output
      
    def compute_cam(self, target_class):
        one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        self.model.zero_grad()
        output.backward(gradient=one_hot_output.to(device), retain_graph=True)

        gradients = self.gradients[-1].cpu().data.numpy()
        feature_maps = self.feature_maps[-1].cpu().data.numpy()

        weights = np.mean(gradients, axis=(2, 3, 4))  # global average pooling
        cam = np.sum(np.multiply(weights, feature_maps), axis=1)

        # Relu
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)

        return cam
    
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

    def reshape_transform(tensor, height=16, width=16):
        print(tensor.shape)
        print(tensor[:, 1:, :].shape)
        result = tensor[:, 1:, :].reshape(tensor.size(0),
                                        height, width, tensor.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        return result
    target_layers = [model.norm]
    
    
    with open(args.img_path, 'rb') as f:
        vr = VideoReader(f, ctx=cpu(0))
    batches = [list(range(i*args.num_frames, (i*args.num_frames)+args.num_frames)) for i in range(len(vr) // args.num_frames)]
    
    for batch_id, frame_id_list in enumerate(batches):
        video_data = vr.get_batch(frame_id_list).asnumpy()
        img = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid, _ in enumerate(frame_id_list)]
        img = [i.resize((224,224)) for i in img]
        transforms = DataAugmentationForVideoMAE(args)
        img = transforms((img, None)) # T*C,H,W

        img = img.view((args.num_frames , 3) + img.size()[-2:]).transpose(0,1) # T*C,H,W -> T,C,H,W -> C,T,H,W
        
        img = img.unsqueeze(0)
        # results = grad_cam(img)
        res = GradCAM(model=model)
        print(res)
    

if __name__ == '__main__':
    opts = get_args()
    main(opts)
