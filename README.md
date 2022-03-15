# CarsAndPlants

## Vehicles and Plants classificator
Neural network model based on EfficientNetV2S trained on [dataset](https://drive.google.com/drive/folders/1wHOf6eGv2esYtqFbBuGW9eigoZhRDmMZ?usp=sharing) which used for vehicles and plants detection on assets catalogue

You can download pretrained model [here](https://drive.google.com/file/d/1s8byGjwD1ziwcyQyRQ1GfPNaT91xokKO/view?usp=sharing)

## Scripts description
- `gen_dataset.py` - dataset preprocessing (generate two files with train and test data)
- `train.py` - training script

## Example of usage
```console
foo@bar:~/CarsAndPlants$ python run.py --model path/to/model/file --dir path/to/dir/with/images
```
