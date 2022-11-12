from applications import load_stn
from utils.splat2d_cuda import splat2d
from utils.laplacian_blending import LaplacianBlender
import torch
from torchvision import transforms
import math
import numpy as np
from PIL import Image
from models import SpatialTransformer


class MyDict(): 
    def __init__(self): pass

def option():
    use_flipping = False
    video_size = 512
    video_path = 'assets/elon/'
    object_picker = 'mouse.png'  #'celeba_moustache.png' #'celeba_pokemon.png'

    args = MyDict()
    args.device = torch.device('cuda')
    args.real_size = video_size
    args.real_data_path = video_path
    args.fps = 30
    args.batch = 1
    args.blend_alg = 'alpha' #'laplacian' #'laplacian_light'
    args.transform = ['similarity', 'flow']
    args.flow_size = 128
    args.stn_channel_multiplier = 0.5
    args.num_heads = 1
    args.distributed = False  # Colab only uses 1 GPU
    args.clustering = False
    args.cluster = None
    args.objects = True
    args.no_flip_inference = not use_flipping
    args.save_frames = False
    args.overlay_congealed = False
    args.ckpt = model = 'celeba'
    args.override = False
    args.save_correspondences = False
    args.out = 'visuals'
    args.label_path = f'assets/objects/{object_picker}'
    args.resolution = 4 * int(video_size)
    args.sigma = 0.3
    args.opacity = 1.0
    args.objects = True
    return args

@torch.inference_mode()
def load_dense_label(path, resolution=None, load_colors=False, device='cuda'):
    """
    This function loads an RGBA image and returns the coordinates of pixels that have a non-zero alpha channel value.
    For augmented reality applications, this function can also return the RGB colors of the image (load_colors=True).
    :param path: Path to the RGBA image file
    :param resolution: Resolution to resize the RGBA image to (default: no resizing)
    :param load_colors: If True, returns (1, P, 3) tensor of RGB values in range [-1, 1] (P = number of coordinates)
    :param device: Device to load points and colors to
    :return: (1, P, 2) tensor of pixel coordinates, (1, P, 3) tensor of corresponding RGB colors, (1, P, 1) tensor of
              corresponding non-zero alpha channel values. The pixel coordinates are stored in (x, y) format and are
              integers in the range [0, W-1] (x coordinates) or [0, Y-1] (y coordinates). RGB values are in [-1, 1] and
              alpha channel values are in [0, 1].
    """
    label = torch.from_numpy(np.asarray(Image.open(path).convert('RGBA'))).to(device)  # (H, W, 4) RGBA format
    label = label[...,[2,1,0,3]]
    label = label.permute(2, 0, 1).unsqueeze_(0)  # (1, 4, H, W)
    if resolution is not None and resolution != label.size(0):  # optionally resize the label:
        label = torch.nn.functional.interpolate(label.float(), scale_factor=resolution / label.size(2), mode='bilinear')
    assert label.size(1) == 4
    i, j = torch.where(label[0, 3] > 0)  # i indexes height, j indexes width
    points = torch.stack([j, i], -1)  # (P, 2); points are stored in (x, y) format
    points = points.unsqueeze_(0)  # (1, P, 2)
    if load_colors:
        image = label.float().div_(255.0)  # (1, 4, H, W)
        alpha_channel = image[:, 3:4, i, j].permute(0, 2, 1)  # (1, P, 1), [0, 1]
        colors = image[:, :3, i, j].add(-0.5).mul(2.0).permute(0, 2, 1)  # (1, P, 3), [-1, 1]
    else:
        alpha_channel = torch.ones(1, points.size(1), 1, device=device, dtype=torch.float)
        colors = None
    return points, colors, alpha_channel

def nchw_center_crop(img):
    # Essentially same as the above function but for (N,C,H,W) PyTorch tensors and doesn't resize (only crops)
    assert img.dim() == 4
    H, W = img.size(2), img.size(3)
    crop = min(H, W)
    top_start = (H - crop) // 2
    left_start = (W - crop) // 2
    img = img[:, :, top_start: (H + crop) // 2, left_start: (W + crop) // 2]
    return img, (top_start, left_start)

def get_plotly_colors(num_points, colorscale):
    color_steps = torch.linspace(start=0, end=1, steps=num_points).tolist()
    colors = get_color(colorscale, color_steps)
    colors = [plotly.colors.unlabel_rgb(color) for color in colors]
    colors = torch.tensor(colors, dtype=torch.float, device='cuda').view(1, num_points, 3)
    colors = colors.div(255.0).add(-0.5).mul(2)  # Map [0, 255] RGB colors to [-1, 1]
    return colors  # (1, P, 3)

def splat_points(images, points, sigma, opacity, colorscale='turbo', colors=None, alpha_channel=None,
                 blend_alg='alpha'):
    """
    Highly efficient GPU-based splatting algorithm. This function is a wrapper for Splat2D to overlay points on images.
    For highest performance, use the colors argument directly instead of colorscale.
    images: (N, C, H, W) tensor in [-1, +1]
    points: (N, P, 2) tensor with values in [0, resolution - 1] (can be sub-pixel/non-integer coordinates)
             Can also be (N, K, P, 2) tensor, in which case points[:, i] gets a unique colorscale
    sigma: either float or (N,) tensor with values > 0, controls the size of the splatted points
    opacity: float in [0, 1], controls the opacity of the splatted points
    colorscale: [Optional] str (or length-K list of str if points is size (N, K, P, 2)) indicating the Plotly colorscale
                 to visualize points with
    colors: [Optional] (N, P, 3) tensor (or (N, K*P, 3)). If specified, colorscale will be ignored. Computing the colorscale
            often takes several orders of magnitude longer than the GPU-based splatting, so pre-computing
            the colors and passing them here instead of using the colorscale argument can provide a significant
            speed-up.
    alpha_channel: [Optional] (N, P, 1) tensor (or (N, K*P, 1)). If specified, colors will be blended into the output
                    image based on the opacity values in alpha_channel (between 0 and 1).
    blend_alg: [Optiona] str. Specifies the blending algorithm to use when merging points into images.
                              Can use alpha compositing ('alpha'), Laplacian Pyramid Blending ('laplacian')
                              or a more conservative version of Laplacian Blending ('laplacian_light')
    :return (N, C, H, W) tensor in [-1, +1] with points splatted onto images
    """
    assert images.dim() == 4  # (N, C, H, W)
    assert points.dim() == 3 or points.dim() == 4  # (N, P, 2) or (N, K, P, 2)
    batch_size = images.size(0)
    if points.dim() == 4:  # each index in the second dimension gets a unique colorscale
        num_points = points.size(2)
        points = points.reshape(points.size(0), points.size(1) * points.size(2), 2)  # (N, K*P, 2)
        if colors is None:
            if isinstance(colorscale, str):
                colorscale = [colorscale]
            assert len(colorscale) == points.size(1)
            colors = torch.cat([get_plotly_colors(num_points, c) for c in colorscale], 1)  # (1, K*P, 3)
            colors = colors.repeat(batch_size, 1, 1)  # (N, K*P, 3)
    elif colors is None:
        num_points = points.size(1)
        if isinstance(colorscale, str):  # All batch elements use the same colorscale
            colors = get_plotly_colors(points.size(1), colorscale).repeat(batch_size, 1, 1)  # (N, P, 3)
        else:  # Each batch element uses its own colorscale
            assert len(colorscale) == batch_size
            colors = torch.cat([get_plotly_colors(num_points, c) for c in colorscale], 0)
    if alpha_channel is None:
        alpha_channel = torch.ones(batch_size, points.size(1), 1, device='cuda')
    if isinstance(sigma, (float, int)):
        sigma = torch.tensor(sigma, device='cuda', dtype=torch.float).view(1).repeat(batch_size)
    blank_img = torch.zeros(batch_size, images.size(1), images.size(2), images.size(3), device='cuda')
    blank_mask = torch.zeros(batch_size, 1, images.size(2), images.size(3), device='cuda')

    prop_obj_img = splat2d(blank_img, points, colors, sigma, False)  # (N, C, H, W)
    prop_mask_img = splat2d(blank_mask, points, alpha_channel, sigma, True) * opacity  # (N, 1, H, W)
    
    if blend_alg == 'alpha':
        out = prop_mask_img * prop_obj_img + (1 - prop_mask_img) * images  # basic alpha-composite
    elif blend_alg == 'laplacian':
        blender = LaplacianBlender().to(images.device)
        out = blender(images, prop_obj_img, prop_mask_img)
    elif blend_alg == 'laplacian_light':
        blender = LaplacianBlender(levels=3, gaussian_kernel_size=11, gaussian_sigma=0.5).to(images.device)
        out = blender(images, prop_obj_img, prop_mask_img)
    return out


def determine_flips(args, t, classifier, input_imgs, cluster=None, return_cluster_assignments=True):
    # There are two ways a flip can be done with GANgealing at test time:
    # (1) For clustering models, directly predict if an input image should be flipped using the cluster classifier net
    # (2) In general, try running both img and flip(img) through the STN. Decide to flip img based on which of the two
    #     produces the smoothest residual flow field.
    # This function predicts the flip using method (1) if classifier is supplied; otherwise uses method (2).
    data_flipped = input_imgs
    flip_indices = torch.zeros(input_imgs.size(0), 1, 1, 1, device=input_imgs.device, dtype=torch.bool)
    warp_policy = 'cartesian'
    clusters = torch.zeros(input_imgs.size(0), dtype=torch.long, device=input_imgs.device)
    return data_flipped, flip_indices, warp_policy, clusters


class GanGealing:
    TRANSFORM = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    ])

    def __init__(self, args):
        print('[*] Loading Options')
        self.args = args
        self.label_path = self.args.label_path
        self.resolution = self.args.resolution
        self.objects = self.args.objects
        self.real_size = self.args.real_size
        self.device = self.args.device
        print('[*] Loading Spatial Transformer')
        self.stn = load_stn(self.args)
        self.prepare()
        print('[*] Warm-Up Spatial Transformer')
        self.warmup()
        print('[*] Ready!')

    def prepare(self):
        self.points, self.colors, self.alpha_channels = load_dense_label(self.label_path, self.resolution, self.objects)
        self.points = SpatialTransformer.normalize(self.points, self.real_size, self.resolution)
        self.classifier = None
        self.mode = 'unimodal'

    def warmup(self):
        # [B=1 3 512 512]
        x = torch.rand(1, 3, 512, 512, device=self.device)
        # [B=1 3 128 128]
        y = self.stn(x)

    def forward(self, batch):
        # [B 3 512 512]
        batch = GanGealing.TRANSFORM(batch)[None]
        batch = batch.to('cuda')
        # B
        B = batch.size(0)
        # Handle cropping if needed:
        frames_are_non_square = batch.size(2) != batch.size(3)
        original_batch = batch
        if frames_are_non_square:
            batch, (y_start, x_start) = nchw_center_crop(batch)  # perform a center crop to make frames square

        # Propagate correspondences to the next batch of video frames:
        batch_flipped, flip_indices, warp_policy, active_cluster_ix = \
            determine_flips(self.args, self.stn, self.classifier, batch, cluster=self.args.cluster, return_cluster_assignments=True)
        points_in = self.points.repeat(B, 1, 1)
        # Perform the actual propagation:
        propagated_points = self.stn.uncongeal_points(batch_flipped, points_in, normalize_input_points=False,  # already normalized above
                                                warp_policy=warp_policy,
                                                padding_mode=self.args.padding_mode, iters=self.args.iters)
        # Flip points where necessary:
        propagated_points[:, :, 0] = torch.where(flip_indices.view(-1, 1),
                                                self.args.real_size - 1 - propagated_points[:, :, 0],
                                                propagated_points[:, :, 0])
        
        # If cropping was performed, we need to adjust our coordinate system to overlay points correctly
        # in the original, uncropped video:
        if frames_are_non_square:
            propagated_points[:, :, 0] += x_start
            propagated_points[:, :, 1] += y_start

        # Select the colorscale for visualization:
        colors_in = self.colors.repeat(B, 1, 1)
        alpha_channels_in = self.alpha_channels.repeat(B, 1, 1)

        # [-1 1] [1 3 512 512]
        out = splat_points(original_batch, propagated_points, sigma=self.args.sigma, opacity=self.args.opacity,
                                    colors=colors_in, alpha_channel=alpha_channels_in, blend_alg=self.args.blend_alg)
        out = (out + 1) / 2.
        out = out * 255
        out = out.clamp_(min=0., max=255.).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        out = out[0]
        return out


if __name__ == '__main__':
    # option
    args = option()
    # model
    gg = GanGealing(args)
    # input
    batch = torch.randint(0, 255, (512, 512, 3)).numpy().astype(np.uint8)
    batch = Image.fromarray(batch)
    # run
    out = gg.forward(batch)
    # check
    assert type(out) == np.ndarray
    assert out.shape == (512, 512, 3)
    assert out.dtype == np.uint8
    print('[*] Done!')