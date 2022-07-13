# NLP_individual_project
## environment
Python 3.7.10
Pytroch 1.8.0
## Data
The penn file's data is from [this](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz)\
Trainset has 42068 sentences Validationset has 3370 sentences Testset has 3761 sentences\
The data file's data is a tiny version of penn,by 10x downsampling.\
Trainset has 4206 sentences Validationset has 337 sentences Testset has 376 sentences\
## Train
```shell
python run.py --mode train [--step ] [--hidden ] [--emb_size] [--batch_size] [--lr] [--train_path] [--valid_path] [--model]
```
## Test
```shell
python run.py --mode test [--batch_size] [--model] [--test_path] [-ckpt]
```
## Result Visualisation
```shell
python draw.py
```
