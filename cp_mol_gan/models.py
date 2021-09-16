import math
from scipy import linalg
import logging

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.layers.merge import _Merge
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.keras.losses import mean_squared_error, categorical_crossentropy
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate, Embedding, concatenate, RNN, GRUCell
from tensorflow.keras.layers import  Activation, Permute, BatchNormalization, Dropout, Add, dot, Dot, Multiply, Subtract 
from tensorflow.keras.layers import TimeDistributed, Bidirectional
from tensorflow.keras.layers import GRU, LSTM
from tensorflow.keras.layers import ELU, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from utils import *


def build_encoder_decoder(num_tokens, latent_dim= 256, weights=None, verbose=True):
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_tokens))
    gru_cells = [GRUCell(hidden_dim) for hidden_dim in [256,256,256]]
    gru_encoder = RNN(gru_cells, return_state=True)
    encoder_outputs_and_states = gru_encoder(encoder_inputs)
    encoder_states = encoder_outputs_and_states[1:]
    encoder_states = Concatenate(axis=1)(encoder_states)

    z_mean = Dense(latent_dim, name='z_mean', activation = 'linear')(encoder_states)
    z_log_var = Dense(latent_dim, name='z_log_var', activation = 'linear')(encoder_states)

    def sampling(args):
        z_mean_, z_log_var_ = args
        batch_size = K.shape(z_mean_)[0]
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev = 1.)
        return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

    z_mean_log_var_output = Concatenate(name='KL')([z_mean, z_log_var])
    sampled_encoder_states = Lambda(sampling, output_shape=(latent_dim,), name='sampled')([z_mean, z_log_var])
    sampled_encoder_states = Activation('tanh')(sampled_encoder_states)
    z_mean_tanh = Activation('tanh')(z_mean)

    def vae_loss(dummy_true, x_mean_log_var_output):
        x_mean, x_log_var = tf.split(x_mean_log_var_output, 2, axis=1)
        kl_loss = - 0.5 * K.mean(1 + x_log_var - K.square(x_mean) - K.exp(x_log_var), axis = -1)
        return kl_loss

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_tokens))

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    Decoder_Dense_Initial = Dense(sum([256,256,256]), activation=None, name = 'Decoder_Dense_Initial')
    decoder_cell_inital = Decoder_Dense_Initial(sampled_encoder_states)
    spliter = Lambda(lambda x: tf.split(x,[256,256,256],axis=-1), name='split')
    decoder_cell_inital = spliter(decoder_cell_inital)

    decoder_cells = [GRUCell(hidden_dim) for hidden_dim in [256,256,256]]
    decoder_gru = RNN(decoder_cells, return_sequences=True, return_state=True)
    decoder_outputs_and_states = decoder_gru(decoder_inputs, initial_state=decoder_cell_inital)
    decoder_outputs = decoder_outputs_and_states[0]
    decoder_outputs = Dropout(0.2)(decoder_outputs)
    Decoder_Dense = Dense(num_tokens, activation='softmax', name = 'Decoder_Dense')
    Decoder_Time_Dense = TimeDistributed(Decoder_Dense, name='reconstruction_layer')
    decoder_outputs = Decoder_Time_Dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    if verbose: model.summary()
    
    if weights is not None:
        model.load_weights(weights)
    model.compile(optimizer='adam', loss = 'categorical_crossentropy')

    # Define sampling models
    #Encoders
    encoder_model = Model(encoder_inputs, z_mean_tanh)
    encoder_model_sampling = Model(encoder_inputs, sampled_encoder_states)
    
    #Transform and decoder
    decoder_states_inputs = Input(shape=(latent_dim,))
    transformed_states = Decoder_Dense_Initial(decoder_states_inputs)
    transform_model = Model(decoder_states_inputs, transformed_states)

    transformed_states_inputs = Input(shape=(sum([256,256,256]),))
    decoder_cell_inital = spliter(transformed_states_inputs)
    decoder_outputs_and_states = decoder_gru(decoder_inputs, initial_state=decoder_cell_inital)
    decoder_states = decoder_outputs_and_states[1:]
    decoder_states = Concatenate(axis=1)(decoder_states)
    decoder_outputs = decoder_outputs_and_states[0]
    decoder_outputs = Decoder_Time_Dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + [transformed_states_inputs],
                          [decoder_outputs] + [decoder_states])

    return model, encoder_model, transform_model, decoder_model


###############################################################################################################
###############################################################################################################
###############################################################################################################


class GANwCondition():
    def __init__(self, latent_dim, noise_dim, lr_g = 0.00005, lr_d= 0.00005, condition_dim=1449, verbose=True):       
        self.condition_dim = condition_dim
        self.latent_condition_size = 256
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim

        optimizer_g = RMSprop(lr_g)
        optimizer_c = RMSprop(lr_d)

        # Build  condition encoder
        self.condition_encoder = self.build_condition_encoder(verbose=verbose)
        
        # Build  and compile classifier 
        self.classifier = self.build_classifier(self.condition_encoder, verbose=verbose)
        self.classifier.compile(loss='binary_crossentropy', optimizer=optimizer_c, metrics=['accuracy'])
        
        # Build  generator for pahse 1
        self.G = self.build_generator(self.condition_encoder, name='Generator Phase 1', verbose=verbose)
               
        # Build  discriminator
        self.D = self.build_discriminator(name='Discriminator', verbose=verbose)
        
        # Build and compile the critic
        self.C = self.build_critic(self.D, name='Critic', verbose=verbose)
        self.C.compile(loss=[self.mean_loss, 'MSE'], loss_weights=[1, 10], optimizer=optimizer_c)

        # Build and compile StackGAN
        self.classifier_weight = K.variable(0.)
        self.GAN = self.build_GAN([self.G], self.D, self.classifier, inputSize=self.noise_dim, name='GAN', verbose=verbose)
        self.GAN.compile(loss=[self.wasserstein_loss, 'binary_crossentropy'], loss_weights=[1,5], optimizer=optimizer_g)
    
    def mean_loss(self, y_true, y_pred):
        return K.mean(y_pred)
    
    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)
    
    
    def build_GAN(self, generators, discriminator, classifier, inputSize, name='Combinded Model', verbose=True):
        # For the combined model we will only train the generators
        discriminator.trainable = False
        classifier.trainable = False
        
        # The generator takes noise and the target label as input
        # and generates the corresponding latente space region with that condition
        noise = h = Input(shape=(inputSize,))
        condition = Input(shape=(self.condition_dim,))
        
        #c_mean  = condition_encoder(condition)
        #condition_sampled, mean_log_var = self.condition_sampler([c_mean, c_log_var])
        
        states = []
        for G in generators:
            h = G([h, condition])
            states.append(h)
        
        # The discriminator takes generated latent space cordinates as input and 
        # determines validity if they correspond to the condition
        valid_unconditioned = []
        for s in states:
            h = discriminator(s)
            valid_unconditioned.append(h)
        
        valid_conditioned = []
        for s in states:
            h = classifier([s, condition])
            valid_conditioned.append(h)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        combined = Model([noise, condition], valid_unconditioned + valid_conditioned)
        if verbose: logging.info('\n' + str(name))
        if verbose: combined.summary()
        
        return combined
    
    
    def build_critic(self, discriminator, name='Critic Model', verbose=True):
        
        real_states = Input(shape=(self.latent_dim,))
        fake_states = Input(shape=(self.latent_dim,))
                
        #########
        # Construct weighted average between real and fake images
        inter_states = Lambda(self.RandomWeightedAverage, name='RandomWeightedAverage')([real_states, fake_states])
        
        # The discriminator takes generated latent space cordinates as input and 
        # determines validity if they correspond to the condition
          
        valid = discriminator(real_states)
        fake = discriminator(fake_states)
        inter = discriminator(inter_states)
                
        sub = Subtract()([fake, valid])
        norm = inter_states = Lambda(self.GradNorm, name='GradNorm')([inter, inter_states])
        
        # output: D(G(Z))-D(X), norm ===(nones, ones)==> Loss: D(G(Z))-D(X)+lmbd*(norm-1)**2  
        critic_model = Model(inputs=[real_states, fake_states], outputs=[sub, norm])
        if verbose: logging.info('\n' + str(name))
        if verbose: critic_model.summary()
        
        return critic_model
    
    
    def build_condition_encoder(self, name='Condition Encoder', verbose=True):

        condition = Input(shape=(self.condition_dim,))
        
        #########

        h_condition = Dense(1024, activation=LeakyReLU(alpha=0.2))(condition)
        h_condition = Dense(512, activation=LeakyReLU(alpha=0.2))(h_condition)
        h_condition = Dense(256, activation=LeakyReLU(alpha=0.2))(h_condition)
        
        #########
        
        condition_encoder = Model(condition, h_condition)

        if verbose: logging.info('\n' + str(name))
        if verbose: condition_encoder.summary()
        
        return (condition_encoder)
    
    
    def build_generator(self, condition_encoder, name='Generator', verbose=True):

        noise = Input(shape=(self.noise_dim,))
        condition = Input(shape=(self.condition_dim,))
        condition_encoder.trainable = False
        
        #########
        
        h_nosie = Dense(512, activation=LeakyReLU(alpha=0.2))(noise)
        h_nosie = Dense(256, activation=LeakyReLU(alpha=0.2))(h_nosie)
        
        #########
        
        h_condition = condition_encoder(condition)
        
        #########
        
        h = concatenate([h_nosie, h_condition])
        h = Dense(256, activation=LeakyReLU(alpha=0.2))(h)
        h = Dense(self.latent_dim, activation='tanh')(h)
        
        #########
        
        generator = Model([noise, condition], h)

        if verbose: logging.info('\n' + str(name))
        if verbose: generator.summary()
        
        return (generator)
    
    
    def build_discriminator(self, name='Discriminator', verbose=True):
        
        states = Input(shape=(self.latent_dim,))
        
        ###########################
        # NO BATCH NORMALIZATION! #
        ###########################
        
        h = Dense(256, activation=LeakyReLU(alpha=0.2))(states)
        h = Dense(256, activation=LeakyReLU(alpha=0.2))(h)
        h = Dropout(rate=0.4)(h)
        
        #########
        
        h_unconditioned = Dense(256, activation=LeakyReLU(alpha=0.2))(h)
        h_unconditioned = Dropout(rate=0.4)(h_unconditioned)
        h_unconditioned = Dense(1, activation=None)(h_unconditioned)
        
        #########
        
        discriminator = Model(states, h_unconditioned)

        if verbose: logging.info('\n' + str(name))
        if verbose: discriminator.summary()

        return discriminator


    def build_classifier(self, condition_encoder, name='Classifier', verbose=True):
        
        states = Input(shape=(self.latent_dim,))
        condition = Input(shape=(self.condition_dim,))
        
        #########
        
        h = Dense(256, activation=LeakyReLU(alpha=0.2))(states)
        h = Dense(256, activation=LeakyReLU(alpha=0.2))(h)
        h = Dropout(rate=0.4)(h)
        
        h_condition = condition_encoder(condition)
        h_condition = Dropout(rate=0.4)(h_condition)
        
        #########

        h_conditioned = concatenate([h, h_condition])
        h_conditioned = Dense(256, activation=LeakyReLU(alpha=0.2))(h_conditioned)
        h_conditioned = Dropout(rate=0.4)(h_conditioned)
        h_conditioned = Dense(1, activation='sigmoid')(h_conditioned)
        
        #########
        
        classifier = Model([states, condition], h_conditioned)

        if verbose: logging.info('\n' + str(name))
        if verbose: classifier.summary()

        return classifier
    
    
