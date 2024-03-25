# FRDiff with DiT

In this codebase, we include AutoFR training code for DiT.

## Install
To install requirements, use following script.
```
conda env create --file environment.yml
```


## AutoFR Training
To run AutoFR training, use following script.

```
python sample.py --lr 5e-3 --wgt 1e-3
```

The trained keyframeset and generated sample results will be saved in directory.  

To use Uniform Keyframeset, uncomment line 137-139 in model.py



