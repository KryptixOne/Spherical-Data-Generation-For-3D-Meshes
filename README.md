# Spherical Data Generation for 3D Meshes

## Overview
This project focuses on generating spherical representations of 3D meshes using data from ShapeNet and Acronym datasets. The method employs hemispherical radial ray-casting to transform the data into a structured spherical format.

## Dataset Representation
The ShapeNet and Acronym datasets are used to create a baseline representation, as shown below:

<p align="center">
  <img src="https://github.com/KryptixOne/Spherical-Data-Generation-For-3D-Meshes/blob/main/Images/DatasetGraspsOnMesh_Acronym.PNG" width="300" />
</p>

To achieve this representation, a sphere is placed around each object, with rays cast from its surface to capture spatial and grasp-related information. Only a hemisphere is utilized for data creation.

<p align="center">
  <img src="https://github.com/KryptixOne/Spherical-Data-Generation-For-3D-Meshes/blob/main/Images/SphereAroundScaledPC.PNG" width="300" />
  <img src="https://github.com/KryptixOne/Spherical-Data-Generation-For-3D-Meshes/blob/main/Images/GraspBreakDown.png" width="300" />
</p>

---

## Resulting Data Representation
The resulting dataset consists of a spherical projection of the 3D meshes, organized into different channels:

### **Channel 1: Spherical Depth Data**
Captures depth values by measuring hit-distance between the enclosing hemisphere and the object.

<p align="center">
  <img src="https://github.com/KryptixOne/Spherical-Data-Generation-For-3D-Meshes/blob/main/Images/CreatedGraspDataImages/Spherical%20Depth%20Data.png" width="500" />
</p>

### **Channel 2: Absolute Grasp Position Data**
Maps absolute grasp positions to the nearest ray in the spherical coordinate system.

<p align="center">
  <img src="https://github.com/KryptixOne/Spherical-Data-Generation-For-3D-Meshes/blob/main/Images/CreatedGraspDataImages/PositionalGripperData.png" width="500" />
</p>

### **Channel 3: Orientation Theta Value**
Represents the inclination angle to determine grasp orientation.

<p align="center">
  <img src="https://github.com/KryptixOne/Spherical-Data-Generation-For-3D-Meshes/blob/main/Images/CreatedGraspDataImages/OreintationThetas.png" width="500" />
</p>

### **Channel 4: Orientation Phi Value**
Encodes the azimuthal angle to determine grasp orientation.

<p align="center">
  <img src="https://github.com/KryptixOne/Spherical-Data-Generation-For-3D-Meshes/blob/main/Images/CreatedGraspDataImages/OrientationPhi.png" width="500" />
</p>

### **Channel 5: Rotational Gamma Value**
Captures rotation along the grasp axis.

<p align="center">
  <img src="https://github.com/KryptixOne/Spherical-Data-Generation-For-3D-Meshes/blob/main/Images/CreatedGraspDataImages/Orientation_Rotation.png" width="500" />
</p>

---

## Gaussian Mixture Model (GMM) Representations
A key goal of this dataset is to model grasp position and orientation using probability distributions. Gaussian Mixture Models (GMMs) are applied to create probabilistic representations.

### **Dense Raw Data**
<p align="center">
  <img src="https://github.com/KryptixOne/Spherical-Data-Generation-For-3D-Meshes/blob/main/Images/GMM_test_results/Dense_raw.PNG" width="500" />
</p>

### **GMM Dense Data - Positional Representation**
<p align="center">
  <img src="https://github.com/KryptixOne/Spherical-Data-Generation-For-3D-Meshes/blob/main/Images/GMM_test_results/DenseData_components.png" width="700" />
</p>

### **GMM Dense Data - 3D Representation**
<p align="center">
  <img src="https://github.com/KryptixOne/Spherical-Data-Generation-For-3D-Meshes/blob/main/Images/GMM_test_results/Dense_3D_data_components.png" width="700" />
</p>

### **Sparse Data - Positional Representation**
<p align="center">
  <img src="https://github.com/KryptixOne/Spherical-Data-Generation-For-3D-Meshes/blob/main/Images/GMM_test_results/sparseData.png" width="500" />
</p>

### **GMM Sparse Data - Positional Representation**
<p align="center">
  <img src="https://github.com/KryptixOne/Spherical-Data-Generation-For-3D-Meshes/blob/main/Images/GMM_test_results/SparseDataGMM.png" width="700" />
</p>

---

## Conclusion
This dataset provides a spherical representation of 3D objects and their grasps, which can be leveraged for grasp planning, robotic manipulation, and probabilistic modeling of grasp distributions. The Gaussian Mixture Model analysis further enhances grasp prediction by capturing underlying probabilistic distributions.

---

## Citation
If you use this dataset in your research, please consider citing this work.

---
