# Codebase for "Generating multivariate time series with COmmon Source CoordInated GAN (COSCI-GAN)"

Authors: Ali Seyfi, Jean-Francois Rajotte, Raymond T. Ng

Reference: Ali Seyfi, Jean-Francois Rajotte, Raymond T. Ng,
"Generating multivariate time series with COmmon Source CoordInated GAN (COSCI-GAN)," 
Neural Information Processing Systems (NeurIPS), 2022.
 
Paper Link: https://openreview.net/pdf?id=RP1CtZhEmR

Code Author: Ali Seyfi

Contact: ali.seyfi.12@gmail.com

This directory contains implementations of COSCI-GAN framework for synthetic multivariate time series data generation
using synthetic and real-world datasets.

-   Sine data: Synthetic
-   EEG data: https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State
-   Stock data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG

You can change the architecture of the generator and discriminator to any arbitrary networks, such as Transformers.

### Command inputs:

-   dataset: data_frame_sine_normal, data_frame_sine_freq_change, data_frame_sine_with_anomaly, EEG_Eye_State_ZeroOne_chop_5best_0, EEG_Eye_State_ZeroOne_chop_5best_1, stock_data_24, 
-   nepochs: Number of training epochs
-   batch_size: Number of samples in each batch
-   nsamples: Length of each time series
-   withCD: Flag for using Central Discriminator
-   LSTMG: Flag for use LSTM network for Generators, if False, the generators will be MLP
-   LSTMD: Flag for use LSTM network for Discriminators, if False, the discriminators will be MLP
-   criterion: 'BCE', 'MSE'
-   glr: Generators' learning rate
-   dlr: Discriminators' learning rate
-   cdlr: Central Discriminator's learning rate
-   Ngroups: Number of channels/features
-   real_data_fraction: Fraction of real data to be used for training COSCI-GAN
-   CD_type: Type of Central Discriminator network, choice between "MLP" and "LSTM"
-   gamma: Gamma parameter controls the trade-off between Diversity and Correlation preservation as described in the paper
-   noise_len: Length of input noise

### Example command

```shell
$ python3 run.py --data_name stock_data_24 --nepochs 100 --batch_size 32
--nsamples 24 --withCD True --LSTMG True --LSTMD True --criterion BCE
--glr 0.001 --dlr 0.001 --cdlr 0.0001 --Ngroups 6 --real_data_fraction 10.0
--CD_type MLP --gamma 5.0 --noise_len 32
```
