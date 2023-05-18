# Environment
python==3.8.15
***
pytorch==1.13.0

scikit learn==1.3.1 

matplotlib==3.5.1

# Usage

```python
conda create -n miao python=3.8.15
conda activate zxs
pip install -r requirement.txt
python Cora.py
python Citeseer.py
python Pubmed.py
```

# Result
| Dataset | Cora | Citeseer | Pubmed |
|:-: | :-: | :-: | :-: |
|Accuracy|0.7900|0.6840|0.7680|
|F1_score|0.7914|0.6440|0.7620|
|AUC|0.96|0.89|0.91|
