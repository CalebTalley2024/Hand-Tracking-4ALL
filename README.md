# Hand-Tracking-4ALL

---

# Requirements

- Conda
- Python 3.7.4
- Camera connected to computer
  
---
# Installation

Create conda environment

  

```sh

conda create --name apply_ai python==3.7.4

conda activate apply_ai

```

  

Clone this repository

  

```sh

git clone https://github.com/CalebTalley2024/Alpha-ASL-4ALL

cd Alpha-ASL-4ALL

```

Install requirements with pip

```sh

pip install -r requirements.txt

```
---
# How to use software

1. collect the data 
	1. run `python data_collect.py`
	2. Your camera should pop up
	3. to change the folder, click '1','2','3','4', or '5' to switch to the 'a', 'b' , 'c' ,'d', or 'e' folder
	4. to take a photo click on the "s" key
	5. Exit out by using `Ctrl+C`
2. train the model
	1. Run `python train.py`
	2. in the terminal, you will be asked if you want to export the model, type 'true' to do so.
	3. Exit out by using `Ctrl+C`
3. test the model
	1. Run `python test.py`
	2. you should see your camera
	3. when your hand is in frame the hand you are holding up(right or left) and the predicted letter will pop up.