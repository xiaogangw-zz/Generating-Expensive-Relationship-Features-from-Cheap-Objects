# Generating-Expensive-Relationship-Features-from-Cheap-Objects
This is the tensorflow implementation for the paper "Generating Expensive Relationship Features from Cheap Objects"

## Getting Started
python version: python-3.5;  cuda version: cuda-10;  tensorflow version: tensorflow-1.13

## Datasets
Directly download processed VG features: [Dataset](https://drive.google.com/file/d/1fA7Qt7zkVtLnZv9Hmh2OHjz6V1HpXMin/view?usp=sharing), unzip and put it in the root directory.

## Evaluate the models
Our pre-trained models can be downloded here: [Models](https://drive.google.com/open?id=11Np9o5dOnxIS1uT-ibyRRHpc9WOXK8xc), unzip and put them in the root directory.  
1. To evaluate the zero-shot and low-shot cases: python eval_lowshot.py  
    --model_path: the checkpoint path of the corresponding model;
    --lowshot_num: 0 and 1, 5, 10, 20 for zero-shot and different low-shot cases.

2. To evaluate the all classes case: python eval_wholedata.py  
    --model_path: the checkpoint path of the corresponding model.

## Train the generation model
1. Train the generation model for all classes or low-shot classes: python main.py --training True    
    --out_dir: name of the directory to save training models;  
    --test_setting: wholedata or lowshot (generate new features for all classes or low-shot classes);  
    --L1_weight: the weight of L1 distance;  
    --gpu: GPU id;  
    --num_predicates: the number of predicate classes: VRD: 70 and VG: 100;  
    --ac_weight: the weight of relation classifier loss;  
    --training: True for training and False for testing;  
    --max_epoch: the maximum training epochs;  
    --batch_size: the training batch size;  
  
2. Generate the new features for all classes or low-shot classes: python main.py --test_setting wholedata(lowshot)  
   We train the generation model for 300 epochs and synthesize 15 times of new data for low-shot features, and same number of original data for all classes features.

3. Train the generation model for zeroshot classes: python main_w2v.py --training True  
    --train_file: the training features for generation model;  
    --random_file: the random features used for generating new zero-shot features;  
    
4. Generate the new features for zeroshot classes: python main_w2v.py  
   We train the generation model for 150 epochs and synthesize 20 times of new data for zero-shot features.

## Train the recognition model
1. Obtain a new relationship recognition model: python train_feature_cls.py  
		--mode: wholedata(train the all classes classifier); or lowshot(train the low-shot and zero-shot classifiers);  
		--lowshot_num: 0 for zero-shot; 1,5,10,20 for low-shot; only used when --mode=lowshot;  
		--lowshot_path: the low-shot and zero-shot indexes fies;  
		--train_path: the new features used for training the recognition model, (generated by step 2 or 4 of last operation);    
		--test_path: the test features used for the recognition model;  
  
## Performance
| VG (%)        | ZShot     |  LShot n=1  | LShot n=5 | LShot n=10  |  LShot n=20  |  ALL @50   |  ALL @100  |
| ------ |:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| Paper     |     19.0     |     20.9     |     24.0     |     27.1     |     30.0     |     63.7     |     64.0     |  
| This repo |     19.1     |     21.9     |     25.1     |     27.9     |     30.5     |     64.1     |     64.4     |  

## Citation
@inproceedings{wang2019generating,  
&nbsp;&nbsp;&nbsp;&nbsp;      author    = {Wang, Xiaogang and Sun, Qianru and ANG, Marcelo and CHUA, Tat-Seng},  
&nbsp;&nbsp;&nbsp;&nbsp;      title     = {Generating expensive relationship features from cheap objects},  
&nbsp;&nbsp;&nbsp;&nbsp;      booktitle = {BMVC},  
&nbsp;&nbsp;&nbsp;&nbsp;      year      = {2019},  
}

## Acknowledgements
Our implementations use the source code from the following repository:  
[vtranse](https://github.com/yangxuntu/vrd)
