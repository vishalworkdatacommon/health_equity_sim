
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import os

def build_vae(data_path='sahie_processed.csv'):
    """
    Builds and trains a Variational Autoencoder (VAE) on the SAHIE data.

    Args:
        data_path (str): The path to the processed data file.
    """
    # Construct the absolute path to the data file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_data_path = os.path.join(script_dir, data_path)
    
    # Load the data
    df = pd.read_csv(full_data_path)
    features = ['year', 'statefips', 'countyfips', 'agecat', 'racecat', 'sexcat', 'iprcat']
    data = df[features].values

    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # VAE architecture
    original_dim = scaled_data.shape[1]
    latent_dim = 2
    intermediate_dim = 32

    # Encoder
    inputs = Input(shape=(original_dim,))
    h = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    # Sampling function
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.0)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # Decoder
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(original_dim)
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    # VAE model
    vae = Model(inputs, x_decoded_mean)

    # VAE loss
    reconstruction_loss = K.mean(K.square(inputs - x_decoded_mean))
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    # Train the VAE
    print("Training the VAE...")
    vae.fit(scaled_data, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
    print("VAE training complete.")

    # Generate synthetic data
    print("\nGenerating synthetic data...")
    n_samples = 1000
    random_latent_vectors = np.random.normal(size=(n_samples, latent_dim))
    
    # Need a separate decoder model to generate data
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)
    
    synthetic_data_scaled = generator.predict(random_latent_vectors)
    synthetic_data = scaler.inverse_transform(synthetic_data_scaled)

    synthetic_df = pd.DataFrame(synthetic_data, columns=features)
    print("Synthetic data generated successfully.")
    print(synthetic_df.head())
    
    synthetic_output_path = os.path.join(script_dir, 'synthetic_sahie_data.csv')
    synthetic_df.to_csv(synthetic_output_path, index=False)
    print(f"\nSynthetic data saved to {synthetic_output_path}")


if __name__ == '__main__':
    build_vae()
