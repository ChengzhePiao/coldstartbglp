## Datasets ##
We introduce four datasets to our experiments, i.e.,  OhioT1DM, ABC4D (NCT02053051), CTR3 (NCT02137512) and REPLACE-BG (NCT02258373).

* **OhioT1DM**: Marling, C.; and Bunescu, R. C. 2020. The OhioT1DM Dataset for Blood Glucose Level Prediction: Update 2020. In KDH@ECAI’20, volume 2675, 71–74
* **ShanghaiT1DM and ShanghaiT2DM**: Zhao, Q.; Zhu, J.; Shen, X.; Lin, C.; Zhang, Y.; Liang, Y.; Cao, B.; Li, J.; Liu, X.; Rao, W.; et al. 2023. Chinese diabetes datasets for data-driven machine learning. Scientific Data, 10(1): 35.
* **ArisesT1DM**: Zhu, T.; Uduku, C.; Li, K.; Herrero, P.; Oliver, N.; and
Georgiou, P. 2022. Enhancing self-management in type 1 diabetes with wearables and deep learning. npj Digital Medicine, 5(1): 78. 

All these datasets except ArisesT1DM can be accessed online by following certain guidelines on their websites. Considering the privacy and ethics, ArisesT1DM should only be accessed by contacting the authors of "Enhancing self-management in type 1 diabetes with wearables and deep learning" and asking for permission.

## Codes ##

### Data Preprocessing

After getting these datasets on the Internet, please run the codes under the folder "/gen_datasets" to preprocess the data. 

### Training and Testing

Our proposed model and all the deep baseline methods are under the folder "/models". Please run "train_test_deep_methods.ipynb" to train and test these deep methods. In term of XGBoost and Linear Regression, please run "train_test_xgb_lr.ipynb". Then, **all our experiments can be reproduced**.

We used the following packages with RTX 3090 Ti to run the codes:
* PyTorch 1.11.0
* Scikit-Learn 1.0.2
* Numpy 1.21.5
* Scipy 1.7.3
* Pandas 1.4.2
