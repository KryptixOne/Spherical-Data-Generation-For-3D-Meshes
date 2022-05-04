Dataset creation using the Acroynm and ShapeNet Datasets. Together, we are capable of forming a grasp/mesh object which is the baseline used for creating this dataset.
![](https://github.com/KryptixOne/Spherical-Data-Generation-For-3D-Meshes/blob/main/Images/DatasetGraspsOnMesh_Acronym.PNG = 250x250)


Resulting Data is a spherical projection of the 3-D meshes.

The Data is organized as follows:

Channel 1: Spherical Depth Data, obtain through simple ray casting

Channel 2: Gripper Position Data, Mapped to nearest ray

Channel 3: Spherical Theta Value --> Used to obtain orientation of grasp

Channel 4: Spherical Phi Value --> Used to obtain orientation of grasp

Channel 5: Rotational Alpha Value --> Used to obtain grasp rotation

# Spherical-Data-Generation-For-3D-Meshes
