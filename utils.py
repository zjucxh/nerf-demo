import os
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union, Callable
from argparser import parse_arguments, read_config_file
import matplotlib.pyplot as plt
import logging 

def load_data(config_data, scene= 'torus', device='cpu'):
    logging.basicConfig(level=logging.DEBUG)

    # Access values of config data
    dataset = config_data['dataset']
    scene = dataset[scene]
    nerf_data = os.path.join(dataset['dataset_dir'], 
                             scene['data'])
    #logging.debug(f'nerf data : {nerf_data}')
    #print(f'nerf data path : {nerf_data}')
    nerf_data = np.load(nerf_data)
    images = np.asarray(nerf_data['images'],dtype=np.float32)
    # Normalize image data 
    eps = 1.0e-3
    if np.max(images, axis=None) > 1.0+eps:
        # assume image pixel values ranges from 0 to 255.0
        images = images/255.0
    #logging.debug('images : {0}'.format(images[0,60,60]))
    #breakpoint()
    # show image with pyplot
    #plt.imshow(images[3])
    #plt.show()
    images = torch.from_numpy(nerf_data['images']).to(device=device, dtype=torch.float32)
    poses = torch.from_numpy(nerf_data['poses']).to(device)
    #logging.debug('poses : {0}'.format(poses))
    focal = torch.from_numpy(nerf_data['focal']).to(device)
    return images, poses, focal 


# Given images, camera poses, focal length
# Convert data to positions and view directions used in nerf
def get_rays(height:int, width: int, 
             focal_length:float, 
             c2w:torch.Tensor
             )-> Tuple[torch.Tensor, torch.Tensor]:
    # Apply pinhole camera model to gather directions at each pixels
    i,j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32).to(c2w),
        torch.arange(height, dtype=torch.float32).to(c2w),
        indexing='ij')
    directions = torch.stack([(i-width * 0.5)/focal_length,
                              -(j-height * 0.5)/focal_length,
                              -torch.ones_like(i)],dim=-1)
    # Apply camear pose to directions
    rays_d = torch.sum(directions[...,None,:] * c2w[:3,:3], dim=-1)
    
    # Origin is the same for all directions (the optical center)
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d
    
# Startified sampling, 
def sample_stratified(
        rays_o:torch.Tensor,
        rays_d:torch.Tensor,
        near:float,
        far:float,
        n_samples:int,
        perturb:Optional[bool]=True,
        inverse_depth:bool=False
)-> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Sample along ray from regularly-spaced bins
    """
    t_vals = torch.linspace(0.0, 1.0, n_samples, device=rays_o.device)
    #print('t_vals : {0}'.format(t_vals))
    if not inverse_depth:
        # Sample linearly between 'near' and 'far'
        z_vals = near * (1.0 - t_vals)  + far * (t_vals)
    else:
        # Sample linearly in invers depth (disparity)
        z_vals = 1.0/(1.0/near * (1.0 - t_vals) + 1.0/far * (t_vals))

    #print('z_vals : {0}'.format(z_vals))

    # Draw uniform samples from bins along ray
    if perturb:
        mids = 0.5 * (z_vals[1:] + z_vals[:-1])
        upper = torch.concat([mids, z_vals[-1:]], dim=-1)
        lower = torch.concat([z_vals[:1], mids], dim=-1)
        t_rand = torch.rand([n_samples], device=z_vals.device)
        z_vals = lower + (upper - lower) * t_rand
    z_vals = z_vals.expand(list(rays_o.shape[:-1]) + [n_samples])
    pts = rays_o[..., None, :] +rays_d[..., None, :] * z_vals[..., :, None]
    return pts, z_vals

class PositionalEncoder(nn.Module):
    r'''
    Sine-cosine positional encoder for input points.
    '''
    def __init__(self, d_input: int, n_freqs: int, log_space:bool=False):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        self.embed_fns = [lambda x : x]

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2.0 **torch.linspace(0.0, self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2.0 ** 0.0, 2.0**(self.n_freqs-1), self.n_freqs)

        # Alternate sin and cosine
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x*freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x*freq))

    def forward(self, x)-> torch.Tensor:
        r"""
        Apply positioanl encoding to input
        """
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)


# Chunks
def get_chunks(inputs:torch.Tensor, \
                chunksize:int=2**15 \
                )->List[torch.Tensor]:
    return [inputs[i:i+chunksize] for i in range(0, inputs.shape[0], chunksize)]


# Prepare chunks
def prepare_chunks(points:torch.Tensor, directions:torch.Tensor, 
                   n_pos_freqs:int, n_dir_freqs:int, 
                   chunk_size:float
                   )->Tuple[List[torch.Tensor], List[torch.Tensor]]:
    pos_encoder = PositionalEncoder(d_input=3, n_freqs=n_pos_freqs, log_space=False)    
    dir_encoder = PositionalEncoder(d_input=3, n_freqs=n_dir_freqs, log_space=False)
    points = points.to(dtype=torch.float32)
    encoded_points = pos_encoder(points)
    directions = points.to(dtype=torch.float32)
    encoded_directions = dir_encoder(directions)
    chunked_points = [encoded_points[i:i+chunk_size] \
                      for i in range(0, encoded_points.shape[0], chunk_size)]
    chunked_directions = [encoded_directions[i:i+chunk_size] \
                          for i in range(0, encoded_directions.shape[0], chunk_size)]
        
    return chunked_points, chunked_directions

def prepare_pos_chunks(points:torch.Tensor,
                       encoding_function:Callable[[torch.Tensor], torch.Tensor],
                       chunksize:int=2**15
                       )->List[torch.Tensor]:
    # Encod and chunkify points for NeRF model
    points = points.reshape(-1,3)
    points = encoding_function(points)
    points = get_chunks(points, chunksize=chunksize)
    return points

def prepare_dir_chunks(points:torch.Tensor, 
                       rays_d:torch.Tensor,
                       encoding_function:Callable[[torch.tensor],torch.Tensor],
                       chunksize:int=2**15
                       )->List[torch.Tensor]:
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    viewdirs = viewdirs[:,None,...].expand(points.shape).reshape((-1,3))
    viewdirs = encoding_function(viewdirs)
    viewdirs = get_chunks(viewdirs, chunksize=chunksize)
    return viewdirs

def cumprod_exclusive(tensor:torch.Tensor)->torch.Tensor:
    cumprod = torch.cumprod(tensor, -1)
    cumprod = torch.roll(cumprod, 1, -1)
    cumprod[...,0] = 1
    return cumprod

# Input raw NeRF output into RGB and other maps
def raw2outputs(raw:torch.Tensor,
                z_vals: torch.Tensor,
                rays_d: torch.Tensor,
                raw_noise_std: float=0.0,
                white_bkgd: bool=False
                )->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Difference between consecutive elements of 'z_vals'. [n_rays, n_samples]
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, 1e10 * torch.ones_like(dists[...,:1])], dim=-1)
    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts fo non-unit directions)
    dists = dists * torch.norm(rays_d[...,None,:],dim=-1)
    # Add noise to model's predictions for density. Can be used to 
    # regularize network during training (prevents floater artifacts)
    noise = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std
    # Predict density of each sample along each ray. Higher values imply
    # higher likelihood of being absorbed at this point. [n_rays, n_samples]
    alpha = 1.0 - torch.exp(-nn.functional.relu(raw[..., 3] + noise) * dists)
    # Compute weight for RGB of each sample along each ray. [n_rays, n_samples]
    # The higher the alpha, the lower subsequent weights are driven.
    weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)

    # Compute weighted RGB map.
    rgb = torch.sigmoid(raw[..., :3])  # [n_rays, n_samples, 3]
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [n_rays, 3]

    # Estimated depth map is predicted distance.
    depth_map = torch.sum(weights * z_vals, dim=-1)

    # Disparity map is inverse depth.
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map),
                              depth_map / torch.sum(weights, -1))

    # Sum of weights along each ray. In [0, 1] up to numerical error.
    acc_map = torch.sum(weights, dim=-1)

    # To composite onto a white background, use the accumulated alpha map.
    if white_bkgd:
      rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map.to(dtype=torch.float32), depth_map, acc_map, weights

# Hierarichal Volume Sampling
def sample_pdf(
        bins: torch.Tensor,
        weights: torch.Tensor,
        n_samples: int,
        perturb: bool=False)->torch.Tensor:
    r'''
    Apply inverse transform sampling to a weighted set of points
    '''
    # Normalize weights to get probability density function(pdf)
    eps = 1.0e-5
    pdf = (weights + eps) / torch.sum(weights + eps, -1, keepdim=True) # [n_rays, weights.shape[-1]]
    # Convert pdf to cdf
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.concat([torch.zeros_like(cdf[...,:1]), cdf], dim=-1) # [n_rays, weight.shape[-1]+1]

    # Take samples positions t grab from CDF. Linear when perturb is 0
    if not perturb:
        u = torch.linspace(0.0, 1.0, n_samples, device=cdf.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples]) # [n_rays, n_samples]
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=cdf.device)
    
    # Find indices along CDF where values in u would be placed
    u = u.contiguous() # Returns contiguous tensor with same values
    inds = torch.searchsorted(cdf, u, right=True) # [n_rays, n_samples]

    # Clamp indices that are out of bounds
    below = torch.clamp(inds-1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1]-1)
    inds_g = torch.stack([below, above], dim=-1) #[n_rays, n_samples, 2]

    #sample from cdf and the corresponding bin centers
    matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1,
                         index=inds_g)
    bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), dim=-1,
                          index=inds_g)

    # Convert samples to ray length.
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples # [n_rays, n_samples]

def sample_hierarchical(
  rays_o: torch.Tensor,
  rays_d: torch.Tensor,
  z_vals: torch.Tensor,
  weights: torch.Tensor,
  n_samples: int,
  perturb: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  r"""
  Apply hierarchical sampling to the rays.
  """

  # Draw samples from PDF using z_vals as bins and weights as probabilities.
  z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
  new_z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], n_samples,
                          perturb=perturb)
  new_z_samples = new_z_samples.detach()

  # Resample points from ray based on PDF.
  z_vals_combined, _ = torch.sort(torch.cat([z_vals, new_z_samples], dim=-1), dim=-1)
  pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :, None]  # [N_rays, N_samples + n_samples, 3]
  return pts, z_vals_combined, new_z_samples

