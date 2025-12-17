# DH-PLR
Partially linear regression (PLR) models provide a powerful framework for inter-
preting key variables while adjusting for complex confounders in modern data
analysis. However, substantial challenges arise for traditional PLR methods
in high-dimensional settings, particularly when both the linear and nonlinear
components involve many predictors. Existing approaches typically tackle high
dimensionality in only one component—either linear or nonlinear—often facing
the curse of dimensionality or a loss of interpretability when both components
are high-dimensional. To overcome these limitations, we propose a Double High-
dimensional Partially Linear Regression (DH-PLR) framework that integrates
the sparsity-inducing SCAD penalty for variable selection in the linear com-
ponent with deep neural networks for flexible nonlinear modeling. Our unified
loss-based formulation accommodates a broad class of loss functions, including
the least-squares, quantile, and Huber loss functions. From a theoretical per-
spective, we establish estimation consistency for both components and variable
selection consistency for the key variables in the linear component. Furthermore,
we develop an efficient algorithm and demonstrate the superior performance of
DH-PLR through simulation studies and an application to breast cancer data.
