import torch
from torch import nn
import os

import numpy as np
import pandas as pd
import pickle as pickle
# import wandb
# from catch22 import catch22_all


class PairwiseDiscriminator(nn.Module):
    def __init__(self, n_channels, alpha):
        super().__init__()
        self.n_channels = n_channels
        n_corr_values = n_channels * (n_channels - 1) // 2
        layers = []
        while np.log2(n_corr_values) > 1:
            layers.append(nn.Linear(n_corr_values, n_corr_values // 2))
            layers.append(nn.LeakyReLU(alpha))
            layers.append(nn.Dropout(0.3))
            n_corr_values = n_corr_values // 2
        layers.append(nn.Linear(n_corr_values, 1))
        layers.append(nn.Sigmoid())
        self.classifier = nn.Sequential(*layers)
        
        self.pairwise_correlation = torch.corrcoef
        self.upper_triangle = lambda x: x[torch.triu(torch.ones(n_channels, n_channels), diagonal=1) == 1]

    def forward(self, x):     
        final_upper_trianle = []
        for i in range(x.shape[0]):
            pairwise_correlation = self.pairwise_correlation(x[i,:].transpose(0,1))
            upper_triangle = self.upper_triangle(pairwise_correlation)
            final_upper_trianle.append(upper_triangle)
        final_upper_trianle = torch.stack(final_upper_trianle)
        return self.classifier(final_upper_trianle)
class LSTMDiscriminator(nn.Module):
    """Discriminator with LSTM"""
    def __init__(self, ts_dim, hidden_dim=256, num_layers=1):
        super(LSTMDiscriminator, self).__init__()
        
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(ts_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1])
        out, _ = self.lstm(x)
        out = self.linear(out.view(x.size(0) * x.size(1), self.hidden_dim))
        out = out.view(x.size(0), x.size(1))
        return out

class Discriminator(nn.Module):
    def __init__(self, n_samples, alpha):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_samples, 256),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(alpha),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            nn.LeakyReLU(alpha),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(alpha),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output

class LSTMGenerator(nn.Module):
    """Generator with LSTM"""
    def __init__(self, latent_dim, ts_dim, hidden_dim=256, num_layers=1):
        super(LSTMGenerator, self).__init__()

        self.ts_dim = ts_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, ts_dim)

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1])
        out, _ = self.lstm(x)
        out = self.linear(out.view(x.size(0) * x.size(1), self.hidden_dim))
        out = out.view(x.size(0), self.ts_dim)
        return out

class Generator(nn.Module):
    def __init__(self, noise_len, n_samples, alpha):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_len, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(alpha),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(alpha),
            nn.Dropout(0.3),
            nn.Linear(512, n_samples)
        )

    def forward(self, x):
        output = self.model(x)
        return output

def COSCIGAN(n_groups,
             id,
             expId,
             dataset,
             num_epochs,
             batch_size,
             n_samples,
             real_data_fraction,
             criterion='BCE',
             with_CD=True,
             LSTM_G=True,
             LSTM_D=True,
             CD_type = 'MLP',
             generator_lr = 0.001,
             discriminator_lr = 0.001,
             central_discriminator_lr = 0.0001,
             gamma_value=5.0,
             noise_len=32):

    name = 'with_CD' if with_CD else 'without_CD'
    temp_name = 'LSTM_G' if LSTM_G else 'MLP_G'
    temp_name += '_LSTM_D' if LSTM_D else '_MLP_D'
    temp_name += '_CD_type_' + CD_type
    full_name = f'{name}_{temp_name}_{n_groups}_{int(real_data_fraction*100)}_{id}_{expId}_{dataset}_{num_epochs}_{batch_size}_{n_samples}_{criterion}_gamma_{gamma_value}_Glr_{generator_lr}_Dlr_{discriminator_lr}_CDlr_{central_discriminator_lr}_noiselen_{noise_len}'

    try:
        if not os.path.isdir('../Results/'):
            os.mkdir('../Results/')
    except:
        pass
    try:
        if not os.path.isdir('../Results/Models/'):
            os.mkdir('../Results/Models/')
    except:
        pass
    try:
        if not os.path.isdir(f'../Results/Models/{name}/'):
            os.mkdir(f'../Results/Models/{name}/')
    except:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(0)

    ##
    try:
        with open('../Dataset/'+dataset+'.csv', 'rb') as fh:
            df = pd.read_csv(fh)
    except:
        with open('../Dataset/'+dataset+'.pkl', 'rb') as fh:
            df = pickle.load(fh)

    ##
    data = df.sample(frac=real_data_fraction).reset_index(drop=True)

    if 'ID' not in data.columns:
        data['ID'] = np.zeros(len(data))

    ##

    train_id = torch.tensor(data['ID'].values.astype(np.float32))
    train = torch.tensor(data.drop('ID', axis = 1).values.astype(np.float32)) 
    train_tensor = torch.utils.data.TensorDataset(train, train_id) 

    ##

    kwargs = {'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        train_tensor, batch_size=batch_size, shuffle=True, **kwargs
    )


    ##
    alpha = 0.1

    ##

    def initialize_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    ##
    discriminators = {}
    if LSTM_D:
        for i in range(n_groups):
            discriminators[i] = LSTMDiscriminator(ts_dim=n_samples)

    else:
        for i in range(n_groups):
            discriminators[i] = Discriminator(n_samples=n_samples, alpha=alpha).apply(initialize_weights)

    for i in range(n_groups):
        discriminators[i] = nn.DataParallel(discriminators[i]).to(device)
        discriminators[i].to(device)  
    ##

    generators = {}
    if LSTM_G:
        for i in range(n_groups):
            generators[i] = LSTMGenerator(latent_dim=noise_len, ts_dim=n_samples)

    else:
        for i in range(n_groups):
            generators[i] = Generator(noise_len=noise_len, n_samples=n_samples, alpha=alpha).apply(initialize_weights)

    for i in range(n_groups):
        generators[i] = nn.DataParallel(generators[i])
        generators[i].to(device)

    ##

    gamma = [gamma_value]*num_epochs

    if criterion == 'BCE':
        loss_function = nn.BCELoss()
    elif criterion == 'MSE':
        loss_function = nn.MSELoss()

    ##

    optimizers_D = {}
    for i in range(n_groups):
        optimizers_D[i] = torch.optim.Adam(discriminators[i].parameters(), lr=discriminator_lr, betas=[0.5, 0.9])

    optimizers_G = {}
    for i in range(n_groups):
        optimizers_G[i] = torch.optim.Adam(generators[i].parameters(), lr=generator_lr, betas=[0.5, 0.9])
    ##

    if with_CD:
        if CD_type == 'LSTM':
            central_discriminator = LSTMDiscriminator(ts_dim=n_samples, num_layers=n_groups)
        elif CD_type == 'MLP':
            central_discriminator = Discriminator(n_samples = n_groups*n_samples, alpha = alpha)
            central_discriminator = central_discriminator.apply(initialize_weights)
        elif CD_type == 'Pairwise':
            central_discriminator = PairwiseDiscriminator(n_channels = n_groups, alpha = alpha)
            central_discriminator = central_discriminator.apply(initialize_weights)
    
        central_discriminator = nn.DataParallel(central_discriminator)
        central_discriminator.to(device)
        optimizer_central_discriminator = torch.optim.Adam(central_discriminator.parameters(), lr=central_discriminator_lr, betas=[0.5, 0.9])

    ##

    for epoch in range(num_epochs):
        for n, (signals, ID) in enumerate(train_loader):
            signals = signals.to(device)
            n_signals = len(signals)

            signal_group = {}
            for i in range(n_groups):
                signal_group[i] = signals[:, i*n_samples:(i+1)*n_samples]

            shared_noise = torch.randn((n_signals, noise_len)).float()

            # Generating samples
            generated_samples = {}
            for i in range(n_groups):
                generated_samples[i] = generators[i](shared_noise).float()

            generated_samples_labels = torch.zeros((n_signals, 1)).to(device).float()
            real_samples_labels = torch.ones((n_signals, 1)).to(device).float()
            all_samples_labels = torch.cat(
                (real_samples_labels, generated_samples_labels)
            )

            # Data for training the discriminators
            all_samples_group = {}
            for i in range(n_groups):
                all_samples_group[i] = torch.cat(
                    (signal_group[i], generated_samples[i])
                )

            # Training the discriminators
            outputs_D = {}
            loss_D = {}
            for i in range(n_groups):
                optimizers_D[i].zero_grad()
                outputs_D[i] = discriminators[i](all_samples_group[i].float())
                loss_D[i] = loss_function(outputs_D[i], all_samples_labels)
                loss_D[i].backward(retain_graph=True)
                optimizers_D[i].step()

            if with_CD:
                # Data from central discriminator
                if CD_type == 'Pairwise':
                    temp_generated = []
                    for i in range(n_groups):
                        temp_generated.append(generated_samples[i])
                    group_generated = torch.stack(temp_generated).transpose(0,1).transpose(1,2)

                    temp_real = []
                    for i in range(n_groups):
                        temp_real.append(signal_group[i])
                    group_real = torch.stack(temp_real).transpose(0,1).transpose(1,2)
                else:
                    temp_generated = generated_samples[0]
                    for i in range(1,n_groups):
                        temp_generated = torch.hstack((temp_generated, generated_samples[i]))
                    group_generated = temp_generated

                    temp_real = signal_group[0]
                    for i in range(1,n_groups):
                        temp_real = torch.hstack((temp_real, signal_group[i]))
                    group_real = temp_real

                all_samples_central = torch.cat((group_generated, group_real))
                all_samples_labels_central = torch.cat(
                    (torch.zeros((n_signals, 1)).to(device).float(), torch.ones((n_signals, 1)).to(device).float())
                )

                # Training the central discriminator
                optimizer_central_discriminator.zero_grad()
                output_central_discriminator = central_discriminator(all_samples_central.float())
                loss_central_discriminator = loss_function(
                    output_central_discriminator, all_samples_labels_central)
                loss_central_discriminator.backward(retain_graph=True)
                optimizer_central_discriminator.step()

            # Training the generators
            outputs_G = {}
            loss_G_local = {}
            loss_G = {}
            for i in range(n_groups):
                optimizers_G[i].zero_grad()
                outputs_G[i] = discriminators[i](generated_samples[i])
                loss_G_local[i] = loss_function(outputs_G[i], real_samples_labels)
                if with_CD:
                    all_samples_central_new = {}
                    output_central_discriminator_new = {}
                    loss_central_discriminator_new = {}

                    generated_samples_new = {}
                    for j in range(n_groups):
                        generated_samples_new[j] = generators[j](shared_noise)

                        if i == j:
                            generated_samples_new[j] = generated_samples_new[j].float()
                        else:
                            generated_samples_new[j] = generated_samples_new[j].detach().float()
 
                    if CD_type == 'Pairwise':
                        temp_generated = []
                        for j in range(n_groups):
                            temp_generated.append(generated_samples_new[j])
                        all_generated_samples = torch.stack(temp_generated).transpose(0,1).transpose(1,2)
                    else:
                        temp_generated = generated_samples_new[0]
                        for j in range(1,n_groups):
                            temp_generated = torch.hstack((temp_generated, generated_samples_new[j]))
                        all_generated_samples = temp_generated
                    
                    all_samples_central_new[i] = torch.cat((all_generated_samples, group_real))   
                    output_central_discriminator_new[i] = central_discriminator(all_samples_central_new[i].float()) 
                    loss_central_discriminator_new[i] = loss_function(
                        output_central_discriminator_new[i], all_samples_labels_central)
                    
                    loss_G[i] = loss_G_local[i] - gamma[epoch] * loss_central_discriminator_new[i]
                else:
                    loss_G[i] = loss_G_local[i]

                loss_G[i].backward(retain_graph=True)
                optimizers_G[i].step()

        log_Dict = {}
        for i in range(n_groups):
            log_Dict[f'loss_D_{i}'] = loss_D[i].cpu()
            log_Dict[f'loss_G_{i}'] = loss_G[i].cpu()
            
        if with_CD:
            log_Dict['loss_CD'] = loss_central_discriminator.cpu()

        if epoch % 10 == 0:
            # Saving the models
            for i in range(n_groups):
                torch.save(generators[i].state_dict(), f'../Results/Models/{name}/{full_name}_Generator_{i}.pt')
                # torch.save(discriminators[i].state_dict(), f'../Results/Models/{name}/{full_name}_Discriminator_{i}.pt')

            # if with_CD:
            #     torch.save(central_discriminator.state_dict(), f'../Results/Models/{name}/{full_name}_Central_Discriminator.pth')


    # Saving the models
    for i in range(n_groups):
        torch.save(generators[i].state_dict(), f'../Results/Models/{name}/{full_name}_Generator_{i}.pt')
        # torch.save(discriminators[i].state_dict(), f'../Results/Models/{name}/{full_name}_Discriminator_{i}.pt')

    # if with_CD:
    #     torch.save(central_discriminator.state_dict(), f'../Results/Models/{name}/{full_name}_Central_Discriminator.pth')


    ########## Post-training results ##########

    # new_noise = torch.randn((1000, noise_len)).float()

    # ## Real data
    # real_samples_final = {}
    # signals_np = {}
    # features_all = {}
    # features_all_df = {}
    # column_names = catch22_all(train[:, :n_samples].detach().numpy()[0])['names']
    # try:
    #     if not os.path.isdir('../../Results/Catch22/'):
    #         os.mkdir('../../Results/Catch22/')
    # except:
    #     pass

    # for i in range(n_groups):
    #     real_samples_final[i] = train[:, i*n_samples:(i+1)*n_samples]
    #     signals_np[i] = real_samples_final[i].detach().numpy()
    #     features_all[i] = np.zeros((signals_np[i].shape[0], 22))

    #     for ind, data in enumerate(signals_np[i]):
    #         features_all[i][ind] = catch22_all(data)['values']

    #     features_all_df[i] = pd.DataFrame(features_all[i], columns=column_names)
    #     features_all_df['ID'] = train_id.numpy().astype(str)
    
    #     # save features_df to csv

    #     features_all_df[i].to_csv(f'../../Results/Catch22/features_df_signal_{i}_real_{dataset}_{str(expId)}.csv')


    # # Generating samples
    # try:
    #     if not os.path.isdir(f'../Results/Catch22/{name}'):
    #         os.mkdir(f'../Results/Catch22/{name}')
    # except:
    #     pass
    # generated_samples_final = {}
    # generated_samples_final_np = {}
    # features_generated_all = {}
    # features_generated_all_df = {}
    # for i in range(n_groups):
    #     generated_samples_final[i] = generators[i](new_noise).cpu().float()
    #     generated_samples_final_np[i] = generated_samples_final[i].detach().numpy()
    #     features_generated_all[i] = np.zeros((generated_samples_final_np[i].shape[0], 22))

    #     for ind, data in enumerate(generated_samples_final_np[i]):
    #         features_generated_all[i][ind] = catch22_all(data)['values']

    #     features_generated_all_df[i] = pd.DataFrame(features_generated_all[i], columns=column_names)

    #     # save features_df to csv

    #     features_generated_all_df[i].to_csv(f'../../Results/Catch22/{name}/features_df_signal_{i}_generated_{full_name}.csv')