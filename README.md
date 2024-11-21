# Vanilla NeRF

Done: convert torus_nerf_data to npz

Done: Test get_rays  

Done: chunkify

Done: given chunked data, nerf forward

Done : raw2output

Done : run on gpu, visualize result

    Done : fix loading data bugs

Done : fix training result is 0 bugs

Done: understanding hierarchical volume sampling

 Hierarchical volume sampling:
 The 3D space is in fact very sparse with acclusions and so most \
 points don't contribute much to the rendered image.
 It is therefore more beneficial to oversample regions with a high \
 likelihood of contributing to the intergral.
 Here we apply learned, normalized weights to the first set
 of samples to create a PDF across the ray, then apply inverse transform \
 sampling to this PDF to gather a second set of samples

WIP:  IBRNET and HumanNeRF

WIP: torus_nerf_data image pixel from [0, 255] to [0.0, 1.0]
