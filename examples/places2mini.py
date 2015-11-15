"""
Runs one epoch of Alexnet on imagenet data.
"""

import sys
from neon.util.argparser import NeonArgparser
from neon.backends import gen_backend
from neon.initializers import Constant, Gaussian
from neon.layers import Conv, Dropout, Pooling, GeneralizedCost, Affine
from neon.optimizers import GradientDescentMomentum, MultiOptimizer, Schedule
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, TopKMisclassification
from neon.models import Model
from neon.data import DataIterator, load_places2_mini
from neon.callbacks.callbacks import Callbacks, Callback

# For running complete alexnet
# alexnet.py -e 90 -val 1 -s <save-path> -w <path-to-saved-batches>
# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()

# hyperparameters
batch_size = 128

(X_train, y_train) = load_places2_mini()

# setup backend
be = gen_backend(backend=args.backend, rng_seed=args.rng_seed, device_id=args.device_id,
                 batch_size=batch_size, default_dtype=args.datatype)

# set up a training set iterator
train_set = DataIterator(X_train, y_train, nclass=100, lshape=(1,128,128))

init1 = Gaussian(scale=0.01)
init1b = Gaussian(scale=0.03)
relu = Rectlin()

# drop LR by 1/250**(1/3) at beginning of epochs 23, 45, 66
weight_sched = Schedule([22, 44, 65], (1/250.)**(1/3.))
opt_gdm = GradientDescentMomentum(0.01, 0.9, wdecay=0.0005, schedule=weight_sched)

# drop bias weights by 1/10 at the beginning of epoch 45.
opt_biases = GradientDescentMomentum(0.02, 0.9, schedule=Schedule([44], 0.1))

# Set up the model layers
layers = [Conv((11, 11, 64), padding=3, strides=4, init=init1, bias=Constant(0), activation=relu),
          Pooling(3, strides=2),
          Conv((5, 5, 192), padding=2, init=init1, bias=Constant(1), activation=relu),
          Pooling(3, strides=2),
          Conv((3, 3, 384), padding=1, init=init1b, bias=Constant(0), activation=relu),
          Conv((3, 3, 256), padding=1, init=init1b, bias=Constant(1), activation=relu),
          Conv((3, 3, 256), padding=1, init=init1b, bias=Constant(1), activation=relu),
          Pooling(3, strides=2),
          Affine(nout=4096, init=init1, bias=Constant(1), activation=relu),
          Dropout(keep=0.5),
          Affine(nout=4096, init=init1, bias=Constant(1), activation=relu),
          Dropout(keep=0.5),
          Affine(nout=100, init=init1, bias=Constant(-7), activation=Softmax())]

cost = GeneralizedCost(costfunc=CrossEntropyMulti())

opt = MultiOptimizer({'default': opt_gdm, 'Bias': opt_biases})

model = Model(layers=layers)

# configure callbacks
callbacks = Callbacks(model, train_set, args, metric=TopKMisclassification(k=5))

model.fit(train_set, optimizer=opt, num_epochs=3, cost=cost, callbacks=callbacks)
