import torch
import numpy as np
import matplotlib.pyplot as plt

from Adversarial_classes.control_action import Control_action
from Adversarial_classes.helper import Helper


class Smoothing:
    def __init__(self, control_action, control_action_perturbed, adv_position, velocity, heading, dt, tar_agent, num_samples_smoothing, sigma_acceleration, sigma_curvature,
                 epsilon_acc_absolute, epsilon_curv_absolute, epsilon_acc_relative, epsilon_curv_relative,
                 pert_model, Domain, T, img, img_m_per_px, num_steps, X, Y, future_action, device):

        # control action
        self.control_action = control_action

        # device 
        self.device = device

        # control actions perturbed
        self.control_action_perturbed = control_action_perturbed

        # Data for dynamical model
        self.data = adv_position
        self.velocity = velocity
        self.heading = heading

        # shapes of data
        self.X = X
        self.Y = Y
        self.future_action = future_action

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
        self.epsilon_acc_relative = epsilon_acc_relative
        self.epsilon_curv_relative = epsilon_curv_relative

        # prediction model settings
        self.pert_model = pert_model
        self.Domain = Domain
        self.T = T
        self.pert_model = pert_model
        self.img = img
        self.img_m_per_px = img_m_per_px
        self.num_steps = num_steps

    def randomized_smoothing(self, smoothing):
        """
        Applies randomized adversarial smoothing and returns the smoothed data.

        Args:
            smoothing (bool): Flag to indicate whether smoothing should be applied.

        Returns:
            tuple: A tuple containing:
                   - X_smoothed (np.ndarray): The smoothed unperturbed data.
                   - X_smoothed_adv (np.ndarray): The smoothed adversarial data.
                   - Y_pred_smoothed (np.ndarray): The predictions for smoothed unperturbed data.
                   - Y_pred_smoothed_adv (np.ndarray): The predictions for smoothed adversarial data.
        """
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
                smoothed_X, Y_Pred_smoothed = self.forward_pass_smoothing(
                    index_sigma, False)

                # append the smoothed unperturbed data
                X_smoothed[index_sigma].append(smoothed_X)
                Y_pred_smoothed[index_sigma].append(Y_Pred_smoothed)

                # smooth adversarial data
                smoothed_X_new, Y_Pred_smoothed_adv = self.forward_pass_smoothing(
                    index_sigma, True)

                # append the smoothed adversarial data
                X_smoothed_adv[index_sigma].append(smoothed_X_new)
                Y_pred_smoothed_adv[index_sigma].append(Y_Pred_smoothed_adv)

        # Return to array
        X_smoothed, X_smoothed_adv, Y_pred_smoothed, Y_pred_smoothed_adv = Helper.convert_to_numpy_array(
            X_smoothed, X_smoothed_adv, Y_pred_smoothed, Y_pred_smoothed_adv)

        return X_smoothed, X_smoothed_adv, Y_pred_smoothed, Y_pred_smoothed_adv

    def forward_pass_smoothing(self, index_sigma, perturbed):
        """
        Performs a forward pass with noise addition and prediction smoothing.

        Args:
            index_sigma (int): Index to select the appropriate sigma values for noise.
            perturbed (bool): Flag to determine if the perturbed control actions should be used.

        Returns:
            tuple: A tuple containing:
                   - data_noised (np.ndarray): The observed data X tensor with added noise, converted to a numpy array.
                   - Y_Pred_smoothed (np.ndarray): The smoothed prediction tensor, converted to a numpy array.
        """

        # Add noise to the target agent
        data_noised = self.add_noise_control_actions(index_sigma, perturbed)

        # Make the prediction and calculate the expectation
        Y_Pred_smoothed = self.pert_model.predict_batch_tensor(X=data_noised,
                                                               T=self.T,
                                                               Domain=self.Domain,
                                                               img=self.img,
                                                               img_m_per_px=self.img_m_per_px,
                                                               num_steps=self.num_steps,
                                                               num_samples=1)

        # Detach tensors
        data_noised = data_noised.detach().cpu().numpy()
        Y_Pred_smoothed = np.squeeze(
            Y_Pred_smoothed.detach().cpu().numpy(), axis=1)

        return data_noised, Y_Pred_smoothed

    def add_noise_control_actions(self, index_sigma, perturbed):
        """
        Adds Gaussian noise to the control actions and applies clamping.

        Args:
            index_sigma (int): Index to select the appropriate sigma values for noise.
            perturbed (bool): Flag to determine if the perturbed control actions should be used.

        Returns:
            torch.Tensor: The data after adding noise to the control actions.
        """
        # Gather gaussian noise for randomized smoothign
        if perturbed:
            control_action_data = self.control_action_perturbed
        else:
            control_action_data = self.control_action

        noise_data = torch.randn_like(control_action_data)

        # Multiply the noise with the predefined sigmas
        noise_data[:, :, :, 0].mul_(self.sigma_acceleration[index_sigma])
        noise_data[:, :, :, 1].mul_(self.sigma_curvature[index_sigma])

        # Remove noise from ego agent
        noise_data[:, 1:] = 0.0

        # apply clamping
        with torch.no_grad():
            # Clamp noise within relative limits
            noise_data[:, :, :,
                       0].clamp_(-self.epsilon_acc_relative, self.epsilon_acc_relative)
            noise_data[:, :, :, 1].clamp_(-self.epsilon_curv_relative,
                                          self.epsilon_curv_relative)

            control_action_noise_data = control_action_data + noise_data

            # Clamp control actions within absolute limits
            control_action_noise_data[:, :, :, 0].clamp_(
                -self.epsilon_acc_absolute, self.epsilon_acc_absolute)
            control_action_noise_data[:, :, :, 1].clamp_(
                -self.epsilon_curv_absolute, self.epsilon_curv_absolute)

        # compute the smoothed data given the noised control actions
        data_noised = Control_action.dynamical_model(
            control_action_noise_data, self.data, self.heading, self.velocity, self.dt, device=self.device)

        data_noised, _ = Helper.return_data(
            data_noised, self.X, self.Y, self.future_action)

        return data_noised
