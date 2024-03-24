# Rethinking Non-Negative Matrix Factorization with Implicit Neural Representations 

### Krishna Subramani<sup>1</sup>, Paris Smaragdis<sup>1</sup>, Takuya Higuchi<sup>2</sup>, Mehrez Souden<sup>2</sup>
### <sup>1</sup>UIUC, <sup>2</sup>Apple

Setup a conda environment using the *environment.yml* file as follows:
~~~
conda env create -f environment.yml
~~~

- All our implicit-neural NMF models are in *models.py*. NMF with multiplicative updates is implemented in *nmf_models.py*.
- *demo_innmf.ipynb* is a jupyter notebook that demonstrates how we train a simple model to factorize a hybrid Time-Frequency representation.
- *nmf_reconstruction.py,nmf_separation.py,innmf_reconstruction.py,innmf_separation.py* contain code to run our experiments on the TIMIT dataset. The *.json* files represent our final results and are plotted in *TIMIT_results.ipynb*.
