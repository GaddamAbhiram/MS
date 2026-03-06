# Adaptive Smoothness-Aware Graph Neural Networks with Attention-Based Feature–Topology Fusion

Author: Abhiram Gaddam

## Setup

```bash
python3.10 -m venv venv310
source venv310/bin/activate
pip install -r requirements.txt
```

## Run Models on Amazon

Default dataset prefix is `./data/amazon/amazon`.

```bash
python scripts/train.py --model asfn
```

Other supported models:

```bash
python scripts/train.py --model gtn
python scripts/train.py --model graphsage
python scripts/train.py --model gcn
python scripts/train.py --model gat
python scripts/train.py --model softgnn
```

If your dataset files are elsewhere, pass the prefix explicitly:

```bash
python scripts/train.py --model asfn --prefix ./data/amazon/amazon
```

