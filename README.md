# Spherical-Data-Generation-For-3D-Meshes
## Spherical Representation of Grasps and ShapeNet Datasets

Using ShapeNet and the Acroynm Datasets (depicted below), we create a spherical representation of the data through hemispherical radial ray-casting

Example ShapeNet + Acroynm baseline representation

<img src="https://github.com/KryptixOne/Spherical-Data-Generation-For-3D-Meshes/blob/main/Images/DatasetGraspsOnMesh_Acronym.PNG" width="300" />



See below an example of sphere enclosing object. Sphere displays ray-casting origin points.
_Note that only a hemisphere is used during data creation_

<img src="https://github.com/KryptixOne/Spherical-Data-Generation-For-3D-Meshes/blob/main/Images/SphereAroundScaledPC.PNG" width="300" />

<img src="https://github.com/KryptixOne/Spherical-Data-Generation-For-3D-Meshes/blob/main/Images/GraspBreakDown.png" width="300" />

## Resulting Data:



### Resulting Data is a spherical projection of the 3-D meshes.

The Data is organized as follows:

**Channel 1:** Spherical Depth Data, obtain through simple ray casting
Image below shows the recording depth/hit-distance between the enclosing hemisphere and the object in question

<img src="https://github.com/KryptixOne/Spherical-Data-Generation-For-3D-Meshes/blob/main/Images/CreatedGraspDataImages/Spherical%20Depth%20Data.png" width="500" />

**Channel 2:** Absolute Grasp Position Data, Mapped to nearest ray
Image below shows the absolute grasp positional data, where points correspond to the nearest ray that a grasp would be located with.

<img src="https://github.com/KryptixOne/Spherical-Data-Generation-For-3D-Meshes/blob/main/Images/CreatedGraspDataImages/PositionalGripperData.png" width="500" />

_Note that for the following orientation data. The values at each plotted position are based on the spherical coordinate system. To identify plane of rotation, 
grasp vector and orgination ray vector are used_

**Channel 3:** Spherical Theta Value --> Used to obtain orientation of grasp

<img src="https://github.com/KryptixOne/Spherical-Data-Generation-For-3D-Meshes/blob/main/Images/CreatedGraspDataImages/OreintationThetas.png" width="500" />

**Channel 4:** Spherical Phi Value --> Used to obtain orientation of grasp

<img src="https://github.com/KryptixOne/Spherical-Data-Generation-For-3D-Meshes/blob/main/Images/CreatedGraspDataImages/OrientationPhi.png" width="500" />

**Channel 5:** Rotational Gamma Value --> Used to obtain grasp rotation

<img src="https://github.com/KryptixOne/Spherical-Data-Generation-For-3D-Meshes/blob/main/Images/CreatedGraspDataImages/Orientation_Rotation.png" width="500" />


### Gaussian Mixture Models (n_components) representations of data:

A primary objective of this dataset is to create a probability mapping of the position and orientation data. To do so, Gaussian mixture models were implemented.

**Dense Raw Data**

<img src="https://github.com/KryptixOne/Spherical-Data-Generation-For-3D-Meshes/blob/main/Images/GMM_test_results/Dense_raw.PNG" width="500" />

**GMM Dense Data Positional**

<img src="https://github.com/KryptixOne/Spherical-Data-Generation-For-3D-Meshes/blob/main/Images/GMM_test_results/DenseData_components.png" width="700" />

**GMM Dense Data 3D**

<img src="https://github.com/KryptixOne/Spherical-Data-Generation-For-3D-Meshes/blob/main/Images/GMM_test_results/Dense_3D_data_components.png" width="700" />

**Sparse Data Positional**

<img src="https://github.com/KryptixOne/Spherical-Data-Generation-For-3D-Meshes/blob/main/Images/GMM_test_results/sparseData.png" width="500" />

**GMM Sparse Data Positional**

<img src="https://github.com/KryptixOne/Spherical-Data-Generation-For-3D-Meshes/blob/main/Images/GMM_test_results/SparseDataGMM.png" width="700" />

