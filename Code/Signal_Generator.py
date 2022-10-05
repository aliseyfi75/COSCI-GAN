import numpy as np
import pandas as pd
import random

def Sine():
    # Parameters
    f_1 = 0.01
    f_2 = 0.005
    sample = 800
    n_samples = 1000

    # Amplitudes
    amp_1 = np.random.normal(0.4, 0.05, n_samples)
    amp_2 = np.random.normal(0.6, 0.05, n_samples)

    # Generate noise
    x = np.arange(sample)
    noise_1 = np.random.normal(0,0.05,(n_samples,sample))
    noise_2 = np.random.normal(0,0.05,(n_samples,sample))

    amp_1 = amp_1.reshape(-1,1)
    amp_2 = amp_2.reshape(-1,1)

    # Generate signal
    sin_1_person_1 = (amp_1*np.sin(2 * np.pi * f_1 * x)) + noise_1
    sin_1_person_2 = (amp_2*np.sin(2 * np.pi * f_1 * x)) + noise_1

    sin_2_person_1 = (amp_1*np.sin(2 * np.pi * f_2 * x)) + noise_2
    sin_2_person_2 = (amp_2*np.sin(2 * np.pi * f_2 * x)) + noise_2

    # Create dataframe
    Sin_1 = np.vstack([sin_1_person_1, sin_1_person_2])
    Sin_2 = np.vstack([sin_2_person_1, sin_2_person_2])

    df_1 = pd.DataFrame(Sin_1)
    df_2 = pd.DataFrame(Sin_2)

    final_df = pd.concat([df_1, df_2], axis=1)

    # Save dataframe
    person_label = np.append(['1']*n_samples, ['2']*n_samples)

    final_df['ID'] = person_label

    final_df.to_pickle("../../Dataset/data_frame_sine_normal.pkl")


def Sine_with_shift():
    # Parameters
    f_1 = 0.01
    f_2 = 0.005
    sample = 800
    n_samples = 1000

    # Amplitudes
    amp_1 = np.random.normal(0.4, 0.05, n_samples)
    amp_2 = np.random.normal(0.6, 0.05, n_samples)

    # Generate noise
    x = np.arange(sample)
    noise_1 = np.random.normal(0,0.05,(n_samples,sample))
    noise_2 = np.random.normal(0,0.05,(n_samples,sample))
    noise_shift_1 = np.random.normal(0,10,n_samples).reshape(-1,1)
    noise_shift_2 = np.random.normal(0,10,n_samples).reshape(-1,1)

    amp_1 = amp_1.reshape(-1,1)
    amp_2 = amp_2.reshape(-1,1)

    # Generate signal
    sin_1_person_1 = (amp_1*np.sin(2 * np.pi * f_1 * x + noise_shift_1)) + noise_1
    sin_1_person_2 = (amp_2*np.sin(2 * np.pi * f_1 * x + noise_shift_1)) + noise_1

    sin_2_person_1 = (amp_1*np.sin(2 * np.pi * f_2 * x + noise_shift_2)) + noise_2
    sin_2_person_2 = (amp_2*np.sin(2 * np.pi * f_2 * x + noise_shift_2)) + noise_2

    # Create dataframe
    Sin_1 = np.vstack([sin_1_person_1, sin_1_person_2])
    Sin_2 = np.vstack([sin_2_person_1, sin_2_person_2])

    df_1 = pd.DataFrame(Sin_1)
    df_2 = pd.DataFrame(Sin_2)

    final_df = pd.concat([df_1, df_2], axis=1)

    # Save dataframe
    person_label = np.append(['1']*n_samples, ['2']*n_samples)

    final_df['ID'] = person_label

    final_df.to_pickle("../../Dataset/data_frame_sine_shift.pkl")

def freq_change():
    # Parameters
    f_1 = np.append([0.01]*400, [0.02]*400)
    f_2 = np.append([0.005]*400, [0.01]*400)
    sample = 800
    n_samples = 1000

    # Amplitudes
    amp_1 = np.random.normal(0.4, 0.05, n_samples)
    amp_2 = np.random.normal(0.6, 0.05, n_samples)

    # Generate noise
    x = np.arange(sample)
    noise_1 = np.random.normal(0,0.05,(n_samples,sample))
    noise_2 = np.random.normal(0,0.05,(n_samples,sample))

    amp_1 = amp_1.reshape(-1,1)
    amp_2 = amp_2.reshape(-1,1)

    # Generate signal
    sin_1_person_1 = amp_1*np.sin(2 * np.pi * f_1 * x) + noise_1
    sin_1_person_2 = amp_2*np.sin(2 * np.pi * f_1 * x) + noise_1

    sin_2_person_1 = amp_1*np.sin(2 * np.pi * f_2 * x)+ noise_2
    sin_2_person_2 = amp_2*np.sin(2 * np.pi * f_2 * x) + noise_2

    # Create dataframe
    Sin_1 = np.vstack([sin_1_person_1, sin_1_person_2])
    Sin_2 = np.vstack([sin_2_person_1, sin_2_person_2])

    df_1 = pd.DataFrame(Sin_1)
    df_2 = pd.DataFrame(Sin_2)

    final_df = pd.concat([df_1, df_2], axis=1)

    # Save dataframe
    person_label = np.append(['1']*n_samples, ['2']*n_samples)

    final_df['ID'] = person_label

    final_df.to_pickle("../../Dataset/data_frame_sine_freq_change.pkl")

def freq_change_twice():
    # Parameters
    f_1 = np.array([0.01]*200 + [0.02]*400 + [0.01]*200)
    f_2 = np.array([0.005]*200 + [0.01]*400 + [0.005]*200)
    sample = 800
    n_samples = 1000

    # Amplitudes
    amp_1 = np.random.normal(0.4, 0.05, n_samples)
    amp_2 = np.random.normal(0.6, 0.05, n_samples)

    # Generate noise
    x = np.arange(sample)
    noise_1 = np.random.normal(0,0.05,(n_samples,sample))
    noise_2 = np.random.normal(0,0.05,(n_samples,sample))

    amp_1 = amp_1.reshape(-1,1)
    amp_2 = amp_2.reshape(-1,1)

    # Generate signal
    sin_1_person_1 = amp_1*np.sin(2 * np.pi * f_1 * x) + noise_1
    sin_1_person_2 = amp_2*np.sin(2 * np.pi * f_1 * x) + noise_1

    sin_2_person_1 = amp_1*np.sin(2 * np.pi * f_2 * x)+ noise_2
    sin_2_person_2 = amp_2*np.sin(2 * np.pi * f_2 * x) + noise_2


    # Create dataframe
    Sin_1 = np.vstack([sin_1_person_1, sin_1_person_2])
    Sin_2 = np.vstack([sin_2_person_1, sin_2_person_2])

    df_1 = pd.DataFrame(Sin_1)
    df_2 = pd.DataFrame(Sin_2)

    final_df = pd.concat([df_1, df_2], axis=1)

    # Save dataframe
    person_label = np.append(['1']*n_samples, ['2']*n_samples)

    final_df['ID'] = person_label

    final_df.to_pickle("../../Dataset/data_frame_sine_freq_change_double.pkl")

def anomaly():
    # Parameters
    f_1 = np.array([0.01]*200 + [0.0]*400 + [0.01]*200)
    f_2 = np.array([0.005]*200 + [0.0]*400 + [0.005]*200)
    sample = 800
    n_samples = 1000

    # Amplitudes
    amp_1 = np.random.normal(0.4, 0.05, n_samples)
    amp_2 = np.random.normal(0.6, 0.05, n_samples)

    # Generate noise
    x = np.arange(sample)
    noise_1 = np.random.normal(0,0.05,(n_samples,sample))
    noise_2 = np.random.normal(0,0.05,(n_samples,sample))

    amp_1 = amp_1.reshape(-1,1)
    amp_2 = amp_2.reshape(-1,1)

    # Generate signal
    sin_1_person_1 = amp_1*np.sin(2 * np.pi * f_1 * x) + noise_1
    sin_1_person_2 = amp_2*np.sin(2 * np.pi * f_1 * x) + noise_1

    sin_2_person_1 = amp_1*np.sin(2 * np.pi * f_2 * x)+ noise_2
    sin_2_person_2 = amp_2*np.sin(2 * np.pi * f_2 * x) + noise_2

    # Create dataframe
    Sin_1 = np.vstack([sin_1_person_1, sin_1_person_2])
    Sin_2 = np.vstack([sin_2_person_1, sin_2_person_2])

    df_1 = pd.DataFrame(Sin_1)
    df_2 = pd.DataFrame(Sin_2)

    final_df = pd.concat([df_1, df_2], axis=1)

    # Save dataframe
    person_label = np.append(['1']*n_samples, ['2']*n_samples)

    final_df['ID'] = person_label

    final_df.to_pickle("../../Dataset/data_frame_sine_with_anomaly.pkl")

def anomaly_more():
    # Parameters
    f_1 = np.array([0.01]*100 + [0]*100 + [0.01]*200 + [0]*200 + [0.01]*200)
    f_2 = np.array([0.005]*100 + [0]*100 + [0.005]*200 + [0]*200 + [0.005]*200)
    sample = 800
    n_samples = 1000

    # Amplitudes
    amp_1 = np.random.normal(0.4, 0.05, n_samples)
    amp_2 = np.random.normal(0.6, 0.05, n_samples)

    # Generate noise
    x = np.arange(sample)
    noise_1 = np.random.normal(0,0.05,(n_samples,sample))
    noise_2 = np.random.normal(0,0.05,(n_samples,sample))

    amp_1 = amp_1.reshape(-1,1)
    amp_2 = amp_2.reshape(-1,1)

    # Generate signal
    sin_1_person_1 = amp_1*np.sin(2 * np.pi * f_1 * x) + noise_1
    sin_1_person_2 = amp_2*np.sin(2 * np.pi * f_1 * x) + noise_1

    sin_2_person_1 = amp_1*np.sin(2 * np.pi * f_2 * x)+ noise_2
    sin_2_person_2 = amp_2*np.sin(2 * np.pi * f_2 * x) + noise_2

    # Create dataframe
    Sin_1 = np.vstack([sin_1_person_1, sin_1_person_2])
    Sin_2 = np.vstack([sin_2_person_1, sin_2_person_2])

    df_1 = pd.DataFrame(Sin_1)
    df_2 = pd.DataFrame(Sin_2)

    final_df = pd.concat([df_1, df_2], axis=1)

    # Save dataframe
    person_label = np.append(['1']*n_samples, ['2']*n_samples)

    final_df['ID'] = person_label

    final_df.to_pickle("../../Dataset/data_frame_sine_with_anomaly_multiple.pkl")


def anomaly_random():
    # Parameters
    f1 = 0.01
    f2 = 0.005
    sample = 800
    n_samples = 1000

    # Amplitudes
    amp_1 = np.random.normal(0.4, 0.05, n_samples)
    amp_2 = np.random.normal(0.6, 0.05, n_samples)

    # Generate noise
    x = np.arange(sample)
    noise_1 = np.random.normal(0,0.05,(n_samples,sample))
    noise_2 = np.random.normal(0,0.05,(n_samples,sample))

    amp_1 = amp_1.reshape(-1,1)
    amp_2 = amp_2.reshape(-1,1)

    # Generate signal
    sin_1_person_1 = np.zeros((n_samples, sample))
    sin_1_person_2 = np.zeros((n_samples, sample))
    sin_2_person_1 = np.zeros((n_samples, sample))
    sin_2_person_2 = np.zeros((n_samples, sample))

    for i in range(n_samples):
        start_1 = random.randrange(150, 250)
        end_1 = random.randrange(150, 250)
        start_2 = random.randrange(150, 250)
        end_2 = random.randrange(150, 250)

        f_1 = np.array([f1]*start_1 + [0]*end_1 + [f1]*(sample-start_1-end_1))
        f_2 = np.array([f2]*start_2 + [0]*end_2 + [f2]*(sample-start_2-end_2))

        sin_1_person_1[i,:] = amp_1[i]*np.sin(2 * np.pi * f_1 * x) + noise_1[i]
        sin_1_person_2[i,:] = amp_2[i]*np.sin(2 * np.pi * f_1 * x) + noise_1[i]

        sin_2_person_1[i,:] = amp_1[i]*np.sin(2 * np.pi * f_2 * x)+ noise_2[i]
        sin_2_person_2[i,:] = amp_2[i]*np.sin(2 * np.pi * f_2 * x) + noise_2[i]

    # Create dataframe
    Sin_1 = np.vstack([sin_1_person_1, sin_1_person_2])
    Sin_2 = np.vstack([sin_2_person_1, sin_2_person_2])

    df_1 = pd.DataFrame(Sin_1)
    df_2 = pd.DataFrame(Sin_2)

    final_df = pd.concat([df_1, df_2], axis=1)

    # Save dataframe
    person_label = np.append(['1']*n_samples, ['2']*n_samples)

    final_df['ID'] = person_label

    final_df.to_pickle("../Dataset/data_frame_sine_with_anomaly_random.pkl")

# Sine_with_shift()
# freq_change()
# freq_change_twice()
# anomaly()
# anomaly_more()
# anomaly_random()