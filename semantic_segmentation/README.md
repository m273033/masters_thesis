# Instructions to reproduce

### 1. Download the datasets:
- CARLA - https://www.kaggle.com/kumaresanmanickavelu/lyft-udacity-challenge
- Cityscapes - https://www.cityscapes-dataset.com/

### 2. Preprocess the Cityscapes dataset's annotations:
Use cityscapes-to-carla.py to convert Cityscapes annotations into CARLA-like annotations. It will reduce the total number of classes in Cityscapes dataset from 30+ classes to 13 classes, which are present in CARLA simulator dataset.

### 3. Split and shuffle the data:
Use shuffle-split.py to shuffle and split the data into train, test and validation sets.

### 4. Combined data:
Use make-combine.py to have a combined dataset

### 5. Instructions to train the model:
- For the combined dataset:
		python3 train_enet.py --dataset_dir="./dataset/combined/" --weighting="ENET" --num_epochs=300 --logdir="./log/combined_run"
- For the model trained only on Cityscapes dataset:
		python3 train_enet.py --dataset_dir="./dataset/cityscapes/" --weighting="ENET" --num_epochs=300 --logdir="./log/cityscapes_run"
- To apply transfer learning from CARLA to Cityscapes:
	- Train the network on CARLA first:
	 python3 train_enet.py --dataset_dir="./dataset/carla/" --weighting="ENET" --num_epochs=300 --logdir="./log/carla_run"
	- Transfer Learning on Cityscapes:
	 python3 train_enet_tl_.py --checkpoint_dir="./log/carla_run" --dataset_dir="./dataset/cityscapes/" --weighting="ENET" --num_epochs=300 --logdir="./log/tl_cityscapes_run" --transfer_learning=True

### 6. Test all of these networks on Cityscapes data:
python3 test_enet.py --dataset_dir="./dataset/cityscapes" --checkpoint_dir="./log/model_to_test" --logdir="./log/model_to_test/test"
