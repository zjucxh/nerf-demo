import logging
import torch
import torch.nn as nn
from argparser import parse_arguments, read_config_file
from utils import PositionalEncoder, sample_hierarchical
from net import NeRF
from typing import List, Tuple
from utils import load_data, sample_stratified, get_rays, prepare_chunks, raw2outputs
from matplotlib import pyplot as plt
import tqdm

#Initialize models, encoders and optimizer for NeRF training
def init_models(config:dict, use_fine_model=True):
    logging.debug(' config data : {0}'.format(config))
    # Positional encoder
    d_input = config['d_input']
    n_pos_freqs = config['n_pos_freqs']     # number of frequencies for points
    n_dir_freqs = config['n_dir_freqs']     # number of frequencies for direction
    log_space = config['log_space']
    pos_encoder = PositionalEncoder(d_input, n_pos_freqs, log_space)
    dir_encoder = PositionalEncoder(d_input, n_dir_freqs, log_space)
    
    skip = tuple(config['skip'])
    logging.debug(' skip : {0}'.format(skip))

    model = NeRF(d_input=pos_encoder.d_output, 
                 n_layers=config['n_layers'],
                 d_filter=config['d_filters'],
                 skip=skip,
                 d_viewdirs=dir_encoder.d_output)
    model.to(config['device'])
    model_params = list(model.parameters())
    if use_fine_model:
        fine_model = NeRF(d_input=pos_encoder.d_output, 
                          n_layers=config['n_layers'],
                          d_filter=config['d_filters'],
                          skip=skip,
                          d_viewdirs=dir_encoder.d_output)
        fine_model = fine_model.to(config['device'])
        model_parmas = model_params + list(fine_model.parameters())
    else:
        fine_model = None
    #print(f'modles :{model}')
    return model, fine_model, pos_encoder, dir_encoder

# Compute forward pass through models
def nerf_forward(rays_o: torch.Tensor,
                 rays_d: torch.Tensor,
                 config: dict,
                 )->Tuple[torch.Tensor, torch.Tensor,torch.Tensor, dict]:
    device = config['device']
    data_object = config['scene']
    dataset = config['dataset']
    n_samples = config['n_samples']
    near = dataset[data_object]['near']
    far = dataset[data_object]['far']
    
    points, z_vals = sample_stratified(rays_o, rays_d, near, far, n_samples)
    return (None)


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO) # Debug level output, WARNING
    args = parse_arguments()
    config_data = read_config_file(args)
    device = config_data['device']
    coarse_model, fine_model, pos_encoder, dir_encoder = init_models(config=config_data)
    scene = config_data['scene']
    dataset = config_data['dataset']
    scene_data = dataset[scene]
    # Load data from config data, return images [N, height, width, channels], poses [N, 4,4]
    # focal length: float
    images, poses, focal_length = load_data(config_data, scene=scene,device=device)
    # Debug: if images loaded correctly
    #print('images data : {0}'.format(images[-1]))
    #breakpoint()
    # Read sampling data from config data
    near = scene_data['near']
    far = scene_data['far']
    n_samples = config_data['n_samples']
    perturb = config_data['perturb'] 
    inverse_depth = config_data['inverse_depth']
    # Read positional encoder configs 
    pe_d_input = config_data['d_input']
    n_pos_freqs = config_data['n_pos_freqs']
    n_dir_freqs = config_data['n_dir_freqs']
    log_space = config_data['log_space']
    # chunkify
    chunk_size = config_data['chunk_size']
    # Hierarchical sampling
    n_hierarchical_sampling = config_data['n_hierarchical_sampling']
    # Optimizer
    lr = config_data['lr']
    use_fine_model = config_data['use_fine_model']
    model_parameters = list(coarse_model.parameters())
    if use_fine_model:
        model_parameters = model_parameters + list(fine_model.parameters())
    
    optimizer = torch.optim.Adam(model_parameters, lr=lr) # If pass fine model, parameters should be changed
    mse_loss = torch.nn.MSELoss()
    #height, width = images.
    logging.debug(' image shape : {0}'.format(images.shape))
    logging.debug(' poses shape : {0}'.format(poses.shape))
    logging.debug(' focal length : {0}'.format(focal_length))
    # get rays and points
    n_images, height, width, n_channels = images.shape
    logging.debug(f' n_image : {n_images}')
    logging.debug(f' height : {height}')
    logging.debug(f' width: {width}')
    logging.debug(f' channels: {n_channels}')
    ## Initialize positional encoder
    #pos_encoder = PositionalEncoder(d_input=pe_d_input,
    #                                n_freqs=n_pos_freqs,
    #                                log_space=log_space)
    #dir_encoder = PositionalEncoder(d_input=pe_d_input,
    #                                n_freqs=n_dir_freqs,
    #                                log_space=log_space)
    n_iters = 0# current training iteration times
    total_ietrs = config_data['n_iters']# total iteration times
    # TODO: plot results for for visualization at certain interval

    test_rgbmap = torch.zeros(width*height,3)
    n_view_image_interval = 500
    t = tqdm.trange(total_ietrs, desc='Training', leave=False)
    for i in t:
        for image_index in range(n_images):
            # compute rays positions and directions, shape : [height, width, 3]
            rays_o, rays_d = get_rays(height, width, focal_length, poses[image_index])
            rays_o = rays_o.reshape(-1,3)
            rays_d = rays_d.reshape(-1,3)
            logging.debug(f' rays position shape : {rays_o.shape}')
            logging.debug(f' rays direction shape : {rays_d.shape}')

            optimizer.zero_grad()

            # Stratified sampling, given rays of positon and direction, 
            # sampling along each ray from near to far
            sampled_points, z_vals = sample_stratified(rays_o, rays_d, 
                                                       near, far, 
                                                       n_samples, perturb,
                                                       inverse_depth)
            logging.debug(" sampled points shape : {0}".format(sampled_points.shape))
            logging.debug(" z_vals shape : {0}".format(z_vals.shape))
            # chunkify and positional encode for sampled points and sampled directions
            sampled_direction = rays_d[...,None,:].expand(sampled_points.shape).reshape(-1,3)
            sampled_points = sampled_points.reshape(-1,3)
            logging.debug(" sampled direction shape : {0}".format(sampled_direction.shape))
            logging.debug(" sampled points shape : {0}".format(sampled_points.shape))
            chunked_pts, chunked_dirs = prepare_chunks(sampled_points, sampled_direction,
                                                       n_pos_freqs,n_dir_freqs,chunk_size)
            logging.debug(' chunked points shape : {0}'.format(len(chunked_pts)))
            logging.debug(' chunked direction shape : {0}'.format(len(chunked_dirs)))
            # Coarse model pass
            predictions = []
            for pts, dirs in zip(chunked_pts, chunked_dirs):
                predictions.append(coarse_model(pts, dirs))
            logging.debug(' len predictions : {0}'.format(len(predictions)))
            nerf_output = torch.cat(predictions, dim=0)
            logging.debug(f' raw_output shape : {nerf_output.shape}')
            logging.debug(f' rays_d shape : {rays_d.shape}')

            # reshape nerf output to [h*w, n_samples, 4]
            nerf_output = nerf_output.view(int(height*width), n_samples, 4)
            rgb_map, depth_map, acc_map, weights = raw2outputs(nerf_output, z_vals, rays_d)
            logging.debug(f' weights shape: {weights.shape}')
            logging.debug(f' rgb map size : {rgb_map.shape}')
            # Fine model pass
            if n_hierarchical_sampling > 0:
                # Backup previous outputs to return 
                rgb_map_0, depth_map_0, acc_map_0 = rgb_map, depth_map, acc_map
                # Apply hierarchical sampling for fine query points
                hierarchical_points, z_vals_combined, z_hierarch = sample_hierarchical(rays_o,
                                                                                rays_d, 
                                                                                z_vals, 
                                                                                weights,
                                                                                n_hierarchical_sampling,
                                                                                perturb)
                
                # Prepare input as before
                hierarchical_directions = rays_d[...,None,:].expand(hierarchical_points.shape).reshape(-1,3)
                hierarchical_points = hierarchical_points.reshape(-1, 3)

                logging.debug(f' hierarchical_directions shape : {hierarchical_directions.shape}')
                #chunked_pts, chunked_dirs = prepare_chunks(query_points, rays_d,)
                logging.debug(f' hierarchical points shape : {hierarchical_points.shape}')
                logging.debug(f' z_val_combined shape: {z_vals_combined.shape}')
                logging.debug(f' z_hierarch shape : {z_hierarch.shape}')
                
                chunked_hierarch_pts, chunked_hierarch_dirs = prepare_chunks(hierarchical_points, hierarchical_directions,
                                                                             n_pos_freqs,n_dir_freqs,chunk_size)
                logging.debug(f' chunded_hierarch_pts shape : {chunked_hierarch_pts[0].shape}')
                logging.debug(f' chunded_hierarch_dirs shape : {chunked_hierarch_dirs[0].shape}')
                predictions = []
                for pts, dirs in zip(chunked_hierarch_pts, chunked_hierarch_dirs):
                    predictions.append(fine_model(pts, dirs))
                nerf_output = torch.cat(predictions, dim=0)
                #breakpoint()
                nerf_output = nerf_output.reshape(int(width*height), (n_samples + n_hierarchical_sampling), 4)
                logging.debug(f' nerf output shape : {nerf_output.shape}')
                rgb_map, depth_map, acc_map, weights = raw2outputs(nerf_output,z_vals_combined, rays_d)
            loss = mse_loss(rgb_map, images[image_index].reshape(-1,3))
            t.set_description('loss : {0}'.format(loss.item()))
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                if n_iters % n_view_image_interval == 0: 
                    test_rgbmap = rgb_map.cpu().detach().clone().numpy().reshape(width, height, 3)
                    plt.imsave('output/{0:0>5}.jpg'.format(n_iters//n_view_image_interval), test_rgbmap)
                    #breakpoint()
                    #print('test rgb map shape : {0}'.format(test_rgbmap.shape))
                    #plt.imshow(test_rgbmap)                    
                    #plt.show()

            n_iters += 1
        
    print('Done')
