import argparse
from Main_modules import COSCIGAN

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=int, default=0)
parser.add_argument('--expId', type=int, default=57)
parser.add_argument('--dataset', type=str, default='stock_data_24')
parser.add_argument('--nepochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--nsamples', type=int, default=24)
parser.add_argument('--withCD', type=bool, default=True)
parser.add_argument('--LSTMG', type=bool, default=True)
parser.add_argument('--LSTMD', type=bool, default=True)
parser.add_argument('--criterion', type=str, default='BCE')
parser.add_argument('--glr', type=float, default=0.001)
parser.add_argument('--dlr', type=float, default=0.001)
parser.add_argument('--cdlr', type=float, default=0.0001)
parser.add_argument('--Ngroups', type=int, default=6)
parser.add_argument('--real_data_fraction', type=float, default=10.0)
parser.add_argument('--CD_type', type=str, default='MLP')
parser.add_argument('--gamma', type=float, default=5.0)
parser.add_argument('--noise_len', type=int, default=32)

id = parser.parse_args().id
expId = parser.parse_args().expId
dataset = parser.parse_args().dataset
num_epochs = parser.parse_args().nepochs
batch_size = parser.parse_args().batch_size
n_samples = parser.parse_args().nsamples
with_CD = parser.parse_args().withCD
LSTM_G = parser.parse_args().LSTMG
LSTM_D = parser.parse_args().LSTMD
criterion = parser.parse_args().criterion
generator_lr = parser.parse_args().glr
discriminator_lr = parser.parse_args().dlr
central_discriminator_lr = parser.parse_args().cdlr
n_groups = parser.parse_args().Ngroups
real_data_fraction = parser.parse_args().real_data_fraction/10
CD_type = parser.parse_args().CD_type
gamma = parser.parse_args().gamma
noise_len = parser.parse_args().noise_len

COSCIGAN(n_groups,
         id,
         expId,
         dataset,
         num_epochs,
         batch_size,
         n_samples,
         real_data_fraction,
         criterion=criterion,
         with_CD=with_CD,
         LSTM_G=LSTM_G,
         LSTM_D=LSTM_D,
         CD_type=CD_type,
         generator_lr=generator_lr,
         discriminator_lr=discriminator_lr,
         central_discriminator_lr=central_discriminator_lr,
         gamma_value=gamma,
         noise_len=noise_len)