import torch
import numpy as np
import matplotlib.pyplot as plt

from Adversarial_classes.control_action import Control_action
from Adversarial_classes.helper import Helper


class Smoothing:
    def __init__(self,dt,tar_agent,num_samples_smoothing,sigma_acceleration,sigma_curvature,
                 epsilon_acc_absolute,epsilon_curv_absolute,control_actions_relative_low,control_actions_relative_high,
                 pert_model,Domain,T,img,img_m_per_px,num_steps):
        # time step
        self.dt = dt

        # tar agent index
        self.tar_agent = tar_agent

        # number of samples for smoothing
        self.num_samples_smoothing = num_samples_smoothing

        # smoothing sigmas
        self.sigma_acceleration = sigma_acceleration
        self.sigma_curvature = sigma_curvature

        # clamping values
        self.epsilon_acc_absolute = epsilon_acc_absolute
        self.epsilon_curv_absolute = epsilon_curv_absolute
        self.control_actions_relative_low = control_actions_relative_low
        self.control_actions_relative_high = control_actions_relative_high

        # prediction model settings
        self.pert_model = pert_model
        self.Domain = Domain
        self.T = T
        self.pert_model = pert_model
        self.img = img
        self.img_m_per_px = img_m_per_px
        self.num_steps = num_steps

    def randomized_smoothing(self, X, X_new, smoothing):
        if smoothing is False:
            return None, None, None, None
        
         # Storage for the inputs with gaussian noise
        X_smoothed = [[] for _ in self.sigma_curvature]
        X_smoothed_adv = [[] for _ in self.sigma_curvature]

        # Storage for the outputs
        Y_pred_smoothed = [[] for _ in self.sigma_curvature]
        Y_pred_smoothed_adv = [[] for _ in self.sigma_curvature]

        # Apply randomized adversarial smoothing
        for index_sigma in range(len(self.sigma_acceleration)):
            for _ in range(self.num_samples_smoothing):
                # smooth unperturbed data
                smoothed_X, Y_Pred_smoothed = self.forward_pass_smoothing(X,index_sigma)

                # append the smoothed unperturbed data
                X_smoothed[index_sigma].append(smoothed_X)
                Y_pred_smoothed[index_sigma].append(Y_Pred_smoothed)

                # smooth adversarial data
                smoothed_X_new, Y_Pred_smoothed_adv = self.forward_pass_smoothing(X_new,index_sigma)

                # append the smoothed adversarial data
                X_smoothed_adv[index_sigma].append(smoothed_X_new)
                Y_pred_smoothed_adv[index_sigma].append(Y_Pred_smoothed_adv)

        # Return to array
        X_smoothed, X_smoothed_adv, Y_pred_smoothed, Y_pred_smoothed_adv = Helper.convert_to_numpy_array(X_smoothed,X_smoothed_adv, Y_pred_smoothed, Y_pred_smoothed_adv)
    
        return X_smoothed, X_smoothed_adv, Y_pred_smoothed, Y_pred_smoothed_adv

    def forward_pass_smoothing(self, data, index_sigma):
        # Add noise to the target agent
        data_noised = self.add_noise_control_actions(data, index_sigma)

        # Make the prediction and calculate the expectation
        Y_Pred_smoothed = self.pert_model.predict_batch_tensor(X=data_noised, 
                                                        T=self.T, 
                                                        Domain=self.Domain, 
                                                        img=self.img, 
                                                        img_m_per_px=self.img_m_per_px, 
                                                        num_steps=self.num_steps, 
                                                        num_samples=1)

        # Detach tensors
        data_noised = data_noised[:,self.tar_agent,:,:].detach().cpu().numpy()
        Y_Pred_smoothed = np.squeeze(Y_Pred_smoothed.detach().cpu().numpy(),axis=1)

        return data_noised, Y_Pred_smoothed

    def add_noise_control_actions(self, data, index_sigma):
        # Retrieve the control actions
        control_action, heading, velocity = Control_action.inverse_Dynamical_Model(data, self.dt)

        # Set the device
        control_action, heading, velocity = Helper.set_device(self.pert_model.device,control_action, heading, velocity)

        # Gather gaussian noise for randomized smoothign
        noise_data = torch.randn_like(control_action)
        
        # Multiply the noise with the predefined sigmas
        noise_data[:, :, :, 0] *= self.sigma_acceleration[index_sigma]
        noise_data[:, :, :, 1] *= self.sigma_curvature[index_sigma]

        # add noise to only target agent
        noise_data[:, 1:] = 0.0
        control_action += noise_data

        with torch.no_grad():
            control_action[:, :, :, 0].clamp_(-self.epsilon_acc_absolute, self.epsilon_acc_absolute)
            control_action[:, :, :, 1].clamp_(-self.epsilon_curv_absolute, self.epsilon_curv_absolute)
            control_action.clamp_(self.control_actions_relative_low, self.control_actions_relative_high)

        data_smoothed = Control_action.dynamical_model(control_action, data, heading, velocity, self.dt)

        return data_smoothed

