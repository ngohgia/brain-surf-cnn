## Generating icosahedral spheres for BrainSurfCNN

The `generate_fslr_icospheres.ipynb` notebook generates the assets needed for the operation on the icosahedral spheres by BrainSurfCNN. The script is a modification from the code by [Jiang et al., 2019 UGSCNN](https://github.com/maxjiang93/ugscnn)

[Libigl](https://libigl.github.io/) and [Human Connectome Project's Workbench](https://www.humanconnectome.org/software/workbench-command) are needed.

## Example

Generate fs_LR spheres using workbench [-surface-create-sphere](https://www.humanconnectome.org/software/workbench-command/-surface-create-sphere).
The spheres are of 100, 200, and 500 vertices respectively.

```
mkdir meshes
wb_command -surface-create-sphere 100 meshes/sphere.100.surf.gii
wb_command -surface-create-sphere 200 meshes/sphere.200.surf.gii
wb_command -surface-create-sphere 500 meshes/sphere.500.surf.gii
```

As an example, the filepaths in `generate_fslr_icospheres.ipynb` notebook are hard-coded. You can change the paths, increase the number of spheres etc. to suit your need.

Running `generate_fslr_icospheres.ipynb` notebook would generate the following files:
- `icosphere_?.pkl`: a dict containing all information of an icosphere and has the same format as that in [Jiang et al., 2019 UGSCNN](https://github.com/maxjiang93/ugscnn).
- `icosphere_?_neighbor_patches.npy`: `Vx7` array with `V` is the number of vertices of an icosphere. Each row of the array consists of the inices of the 7 neighbors of each vertex.
- `icosphere_[i]_to_icosphere_[i+1]_vertices.npy`: contains the indices of the vertices in the next-resolution sphere that are closest to each vertex in the current icosphere. For example, given 2 spheres of 100 (icosphere0) and 200 (icosphere1) vertices, the `icosphere_0_to_icosphere_1_vertices.npy` has 100 indices in icosphere1 that are closest to the vertices in icosphere0.

`icosphere_?_neighbor_patches.npy` and `icosphere_[i]_to_icosphere_[i+1]_vertices.npy` are used for downsampling and upsampling functions.
