# GRU RNN
An minimal and elaborately commented implementation of a recurrent neural network with GRUs ([Gated Recurrent Units, Cho et al.](https://en.wikipedia.org/wiki/Gated_recurrent_unit)) applied to predict the next character in a document given a series of preceding characters in a similar way as [Andrej Karpathy's minimal ordinary RNN implementation](https://github.com/weixsong/min-char-rnn/blob/master/min-char-rnn.py).

To run this implementation an install of Python 3.x and Numpy is required. Running the `main.py` script will start adapting the model to sequences in `input.txt` through backpropagation. Once every 100 model updates the model will sample and print a piece of fully predicted text.

The standard `input.txt` contains [40 paragraphs of lorem ipsum](http://loripsum.net/api/40/verylong/plaintext). After training, the resulting model produces fun pseudo-Latin.
