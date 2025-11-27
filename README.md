# Classfication-Tasks-with-Noise

## Set up Conda Environment

python --version: Python 3.10.19

pip install -r requirements.txt

## Run Code

Run _mnist-noise.py_ 
- parser.add_argument("--noise-train", action="store_true", default=False,
                        help="enable noise model during training")      # Set False if not using noise model in training
- parser.add_argument("--noise-infer", action="store_true", default=True,
                        help="enable noise model during inference")     # Set True if using noise model in valid and test
