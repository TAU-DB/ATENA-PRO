## Setup
Clone the repository and install requirements using the requirements.txt file.
</br> 
ANON-SYS was tested on Conda environment, using Python 3.7.3

## Getting Started
Example command line for training ANON-SYS:
```
$ python train.py --query Q2.txt --env ATENAPROcont-v0 --schema NETFLIX --dataset-number 0 --algo chainerrl_ppo --arch FFParamSoftmax --episode-length 7 --steps 1500000 --eval-interval 100000 --num-envs 64
```
Run `train.py --help` for further options and documentation. <br/>
The `--query` parameter should receive a path to a text file containing the DXL query under the queries directory in this repo.
You can use one of the predefined queries or create new one.

After training, you will get an output directory similar to this: `results/20220501T123456.718192/1500000_finish`
For testing your model, run `test.py` with the mentioned path and the same parameters from training.
In our example, the command line should be:
```
$ python test.py --load results/20220501T123456.718192/1500000_finish --env ATENAPROcont-v0 --schema NETFLIX --dataset-number 0 --algo chainerrl_ppo --arch FFParamSoftmax --episode-length 7
``` 
The output will be an auto-generated EDA session that fits the dataset and the DXL query.

## Adding New Scheme
Adding new scheme to ANON-SYS is a simple process:
1. Upload one or multiple datasets in tsv format to the repository
2. <p>Create two new files:<br/>
    <code>columns_data.py</code> - Holds basic definitions about the data structure.<br/>
    <code>scheme_helpers.py</code> - The logic of parsing the datasets. Probably has only a few changes from other scheme helpers.
    </p>
3. Add option to configure and apply it in `arguments.py` and `global_env_prop.py` 

 


