# Systematic Feature Design for Cycle Life Prediction of Lithium-Ion Batteries During Formation

This repository contains the software for [Systematic Feature Design for Cycle Life Prediction of Lithium-Ion Batteries During Formation] which can be used for designing predictive yet interpretable features for cycle life prediction based only on experimental data during the formation step.

![framework_v4](https://github.com/user-attachments/assets/af8c1326-84ac-4f3c-b7e3-1031835772b9)

# Code

"feature_design_nested_CV.m" is a MATLAB code that demonstrates the overall feature design framework. For the given input data, it overgoes (1) determination of lambda, (2) partitioning based on beta, (3) design of features, (4) merge sections, and (5) feature down-selection.
"figures.m" is a MATLAB code that generates key figures used in the manuscript.

# Folders

"data_formation" folder contains raw measurements of Q^A(V), t^A(V), Q^B(V), V^B(t), Q^C(V), and V^C(t) that are used as the input data candidates.
"tsfresh_feature_extraction" folder contains code for extracting tsfresh autoML features from the measurements in the "data_formation" folder.
"SPA" folder contains the Smart Process Analytics (SPA) Python codes for constructing various agnostic, autoML, and ML models using designed features.
A set of agnostic and autoML features can be found at Zenodo at 10.5281/zenodo.14916092.
"fused_LASSO" folder contains R codes that perform fused-lasso. The input to the code is the input data stored in "data_formation" folder, and the output is the fused-lasso regression coefficient beta's at various lambda values.
"designed_features" folder contains feature matrices that are designed when using Q^B(V) as the input data. Training set, test set, and group labels are provided for each fold in the outer loop.

# Files
"agnostic_results.csv", "autoML_results.csv", and "designed_results.csv" show the median and maximum RMSE and MAPE among five-fold outer loop for every agnostic models, autoML models, and ML models with designed features for every models constructed in this study, respectively.
"prediction_results.xlsx" summarizes the predictive performance of three models at each formation protocol.

## Acknowledgement

This work was supported by the Toyota Research Institute through D3BATT: Center for Data-Driven Design of Li-Ion Batteries.
