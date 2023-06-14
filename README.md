# HAGNN
source code of Hybrid Aggregation for Heterogeneous Graph Neural Networks
# Quick start
```
pip install -r requirements.txt 
```
The dataset can be obtain in [HGB](https://www.biendata.xyz/hgb/#/datasets).
After obtaining data sets, you can enter the NC or LP folder to perform node classification and link prediction tasks.
For node classification task, run
```
cd NC
python run_new.py --dataset DBLP
```
For link prediction tasks, run
```
cd LP
python run_dist.py --dataset PubMed
```
