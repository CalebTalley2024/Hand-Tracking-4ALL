# Alpha-ASL-4ALL
---
# Requirements
- Conda
- Python 3.7.4

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

# Next to do

- Use both left and right to sign the alphabet when taking pictures
## Next
- now we have to
  - 1:  tone hyperparams(since the model is overfit to the training data) 
  - 2: export the model
  - 3: get our model to work in test.py
  - 4: get rid of allowing popup when training the model
  - 5: if need, add git_ignore file so git ignores the image files
  - 6: if the accuracy isnt high enough, use 3 channels of gray scale