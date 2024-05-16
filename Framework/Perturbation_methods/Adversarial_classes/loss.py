from abc import ABC, abstractmethod
import torch
        
# Abstract base class for loss functions
class LossFunction(ABC):
    @abstractmethod
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        pass

# Abstract base class for barrier functions
class BarrierFunction(ABC):
    @abstractmethod
    def calculate_barrier(self, X_new, X, tar_agent):
        pass


# Context for calculating loss with optional barrier
class LossContext:
    def __init__(self, loss_strategy: LossFunction, barrier_strategy: BarrierFunction = None):
        self.loss_strategy = loss_strategy
        self.barrier_strategy = barrier_strategy

    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        loss = self.loss_strategy.calculate_loss(X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent)
        if self.barrier_strategy:
            barrier_output = self.barrier_strategy.calculate_barrier(X_new, X, tar_agent)
            loss -= barrier_output
        return loss

# Static class containing various loss functions and barrier functions
class Loss:
    @staticmethod
    def calculate_loss(X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1, distance_threshold, log_value, spline_data, loss_function, barrier_function, tar_agent, ego_agent):
        """
        Calculates the loss based on the specified loss and barrier functions.

        Args:
            X (torch.Tensor): The ground truth postition tensor with array shape (batch size, number agents, number time steps observed, coordinates (x,y))
            X_new (torch.Tensor): The perturbed position tensor with array shape (batch size, number agents, number time steps observed, coordinates (x,y))
            Y (torch.Tensor): The ground truth future position tensor with array shape (batch size, number agents, number time steps future, coordinates (x,y))
            Y_new (torch.Tensor): The perturbed future position tensor with array shape (batch size, number agents, number time steps future, coordinates (x,y))
            Y_Pred (torch.Tensor): The predicted future position tensor with array shape (batch size, number predictions (K), number time steps future, coordinates (x,y))
            Y_Pred_iter_1 (torch.Tensor): The predicted future position tensor from the first iteration with array shape (batch size, number predictions (K), number time steps future, coordinates (x,y))
            distance_threshold (float): The distance threshold for the barrier function.
            log_value (float): The logarithm base value for the barrier function.
            spline_data (torch.Tensor): The spline data for the barrier function with array shape (batch size, number agents, number spline data, coordinates (x,y))
            loss_function (str): The name of the loss function.
            barrier_function (str): The name of the barrier function.
            tar_agent (int): The index of the target agent.
            ego_agent (int): The index of the ego agent.

        Returns:
            torch.Tensor: The calculated loss.
        """
        loss_function = get_name.loss_function_name(loss_function)
        barrier_function = get_name.barrier_function_name(barrier_function, distance_threshold, log_value, spline_data) if barrier_function else None
        loss_context = LossContext(loss_function, barrier_function)
        return loss_context.calculate_loss(X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1, tar_agent, ego_agent)

    @staticmethod
    def ADE_loss_Y_GT_and_Y_pred(Y, Pred_t, tar_agent):
        """
        Calculates the Average Displacement Error between tar agent's loss between ground truth future positions and predicted positions.

        Args:
            Y (torch.Tensor): The ground truth future position tensor.
            Pred_t (torch.Tensor): The predicted future position tensor.
            tar_agent (int): The index of the target agent.

        Returns:
            torch.Tensor: The ADE loss.
        """
        # norm is over the positions -> inner torch.mean is over the time steps -> outer torch.mean is over the number of prediciton
        return torch.mean(torch.mean(torch.linalg.norm(Y[:,tar_agent,:,:].unsqueeze(1) - Pred_t, dim=-1 , ord = 2), dim=-1),dim=-1) 

    @staticmethod
    def ADE_loss_Y_pred_and_Y_perturb(Y_new, Pred_t, tar_agent):
        """
        Calculates the ADE loss between tar agent's perturbed future positions and predicted positions.

        Args:
            Y_new (torch.Tensor): The perturbed future position tensor.
            Pred_t (torch.Tensor): The predicted future position tensor.
            tar_agent (int): The index of the target agent.

        Returns:
            torch.Tensor: The ADE loss.
        """
        # norm is over the positions -> inner torch.mean is over the time steps -> outer torch.mean is over the number of prediciton
        return torch.mean(torch.mean(torch.linalg.norm(Y_new[:,tar_agent,:,:].unsqueeze(1) - Pred_t, dim=-1 , ord = 2), dim=-1),dim=-1)

    @staticmethod
    def ADE_loss_Y_pred_iteration_1_and_Y_perturb(Y_new, Pred_iter_1, tar_agent):
        """
        Calculates the ADE loss between tar agent's first iteration predicted positions and perturbed future positions.

        Args:
            Y_new (torch.Tensor): The perturbed future position tensor.
            Pred_iter_1 (torch.Tensor): The predicted future position tensor from the first iteration.
            tar_agent (int): The index of the target agent.

        Returns:
            torch.Tensor: The ADE loss.
        """
        # norm is over the positions -> inner torch.mean is over the time steps -> outer torch.mean is over the number of prediciton
        return torch.mean(torch.mean(torch.linalg.norm(Y_new[:,tar_agent,:,:].unsqueeze(1) - Pred_iter_1, dim=-1 , ord = 2), dim=-1),dim=-1)

    @staticmethod
    def ADE_loss_Y_pred_and_Y_pred_iteration_1(Pred_t, Pred_iter_1):
        """
        Calculates the ADE loss between tar agent's current predicted positions and the first iteration predicted positions.

        Args:
            Pred_t (torch.Tensor): The current predicted future position tensor.
            Pred_iter_1 (torch.Tensor): The predicted future position tensor from the first iteration.

        Returns:
            torch.Tensor: The ADE loss.
        """
        # norm is over the positions -> inner torch.mean is over the time steps -> outer torch.mean is over the number of prediciton
        return torch.mean(torch.mean(torch.linalg.norm(Pred_t - Pred_iter_1, dim=-1, ord=2), dim=-1), dim=-1)

    @staticmethod
    def ADE_loss_Y_perturb_and_Y_GT(Y_new, Y, tar_agent):
        """
        Calculates the ADE loss between tar agent's perturbed future positions and ground truth future positions.

        Args:
            Y_new (torch.Tensor): The perturbed future position tensor.
            Y (torch.Tensor): The ground truth future position tensor.
            tar_agent (int): The index of the target agent.

        Returns:
            torch.Tensor: The ADE loss.
        """
        # index 0 is the target agent -> norm is over the positions -> torch.mean is over the time steps 
        return torch.mean(torch.linalg.norm(Y_new[:,tar_agent,:,:] - Y[:,tar_agent,:,:], dim=-1, ord=2), dim=-1)

    @staticmethod
    def collision_loss_Y_ego_GT_and_Y_pred_tar(Y, Pred_t, ego_agent):
        """
        Calculates the collision loss between ego agent's ground truth positions and target agent's predicted positions.

        Args:
            Y (torch.Tensor): The ground truth position tensor.
            Pred_t (torch.Tensor): The predicted position tensor.
            ego_agent (int): The index of the ego agent.

        Returns:
            torch.Tensor: The collision loss.
        """
        #norm is over the positions -> torch.min is over the time steps -> torch.mean is over the number of prediciton
        return torch.mean(torch.linalg.norm(Y[:,ego_agent,:,:].unsqueeze(1) - Pred_t, dim=-1, ord=2).min(dim=-1).values, dim=-1)

    @staticmethod
    def collision_loss_Y_ego_GT_and_Y_perturb_tar(Y_new, Y, tar_agent, ego_agent):
        """
        Calculates the collision loss between ego agent's ground truth positions and target agent's perturbed positions.

        Args:
            Y_new (torch.Tensor): The perturbed position tensor.
            Y (torch.Tensor): The ground truth position tensor.
            tar_agent (int): The index of the target agent.
            ego_agent (int): The index of the ego agent.

        Returns:
            torch.Tensor: The collision loss.
        """
        # norm is over the positions -> torch.min is over the time steps 
        return torch.linalg.norm(Y[:,ego_agent,:,:] - Y_new[:,tar_agent,:,:], dim=-1, ord=2).min(dim=-1).values

    @staticmethod
    def barrier_log_function(distance_threshold, X_new, X, log_value, tar_agent):
        """
        Calculates the barrier log function based on the distance between tar agent's adversarial and original positions.

        Args:
            distance_threshold (float): The distance threshold for the barrier function.
            X_new (torch.Tensor): The adversarial position tensor.
            X (torch.Tensor): The original position tensor.
            log_value (float): The logarithm base value for the barrier function.
            tar_agent (int): The index of the target agent.

        Returns:
            torch.Tensor: The barrier log function value.
        """
        barrier_norm = torch.norm(X_new[:,tar_agent,:,:] - X[:,tar_agent,:,:], dim=-1)
        barrier_log = torch.log(distance_threshold - barrier_norm)
        barrier_log_new = barrier_log / torch.log(torch.tensor(log_value))
        return torch.mean(barrier_log_new, dim=-1)

    @staticmethod
    def barrier_log_function_spline(distance_threshold, X_new, spline_data, log_value, tar_agent):
        """
        Calculates the barrier log function based on the distance between adversarial positions and spline data.

        Args:
            distance_threshold (float): The distance threshold for the barrier function.
            X_new (torch.Tensor): The adversarial position tensor.
            spline_data (torch.Tensor): The spline data tensor.
            log_value (float): The logarithm base value for the barrier function.
            tar_agent (int): The index of the target agent.

        Returns:
            torch.Tensor: The barrier log function value.
        """
        # Calculate the distance between the adversarial observed states and spline data
        distance = torch.cdist(X_new[:,tar_agent,:,:], spline_data[:,tar_agent,:,:], p=2)
        min_indices = torch.argmin(distance, dim=-1)

        # Expand min_indices to match the shape required for gather
        expanded_min_indices = min_indices.unsqueeze(-1).expand(-1, -1, spline_data.size(-1))
        closest_points = torch.gather(spline_data[:,tar_agent,:,:], 1, expanded_min_indices)

        # calculate the barrier function
        barrier_norm = torch.norm(X_new[:,tar_agent,:,:] - closest_points, dim=-1)
        barrier_log = torch.log(distance_threshold - barrier_norm)
        barrier_log_new = barrier_log / torch.log(torch.tensor(log_value))
        return torch.mean(barrier_log_new, dim=-1)

# Helper class for retrieving loss and barrier function instances by name
class get_name:
    @staticmethod
    def loss_function_name(loss_function):
        if loss_function == 'ADE':
            return ADELoss()
        elif loss_function == 'ADE_new_GT':
            return ADELossNewGT()
        elif loss_function == 'ADE_new_pred':
            return ADELossNewPred()
        elif loss_function == 'Collision':
            return CollisionLoss()
        elif loss_function == 'Fake_collision_GT':
            return FakeCollisionLossGT()
        elif loss_function == 'Fake_collision_Pred':
            return FakeCollisionLossPred()
        elif loss_function == 'Hide_collision_GT':
            return HideCollisionLossGT()
        elif loss_function == 'Hide_collision_Pred':
            return HideCollisionLossPred()
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")

    @staticmethod
    def barrier_function_name(barrier_function, distance_threshold, log_value, spline_data):
        if barrier_function == 'Log':
            return LogBarrier(distance_threshold, log_value)
        elif barrier_function == 'Spline':
            return SplineBarrier(distance_threshold, spline_data, log_value)
        else:
            raise ValueError(f"Unknown barrier function: {barrier_function}")

# Specific loss function implementations
class ADELoss(LossFunction):
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        return -Loss.ADE_loss_Y_GT_and_Y_pred(Y, Pred_t, tar_agent)

class ADELossNewGT(LossFunction):
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        return -Loss.ADE_loss_Y_pred_and_Y_perturb(Y_new, Pred_t, tar_agent) + Loss.ADE_loss_Y_perturb_and_Y_GT(Y_new, Y, tar_agent)

class ADELossNewPred(LossFunction):
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        return -Loss.ADE_loss_Y_pred_and_Y_perturb(Y_new, Pred_t, tar_agent) + Loss.ADE_loss_Y_pred_iteration_1_and_Y_perturb(Y_new, Pred_iter_1, tar_agent)

class CollisionLoss(LossFunction):
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        return Loss.collision_loss_Y_ego_GT_and_Y_pred_tar(Y, Pred_t, ego_agent)

class FakeCollisionLossGT(LossFunction):
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        return Loss.collision_loss_Y_ego_GT_and_Y_pred_tar(Y, Pred_t, ego_agent) + Loss.ADE_loss_Y_perturb_and_Y_GT(Y_new, Y, tar_agent)

class FakeCollisionLossPred(LossFunction):
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        return Loss.collision_loss_Y_ego_GT_and_Y_pred_tar(Y, Pred_t, ego_agent) + Loss.ADE_loss_Y_pred_iteration_1_and_Y_perturb(Y_new, Pred_iter_1, tar_agent)

class HideCollisionLossGT(LossFunction):
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        return Loss.collision_loss_Y_ego_GT_and_Y_perturb_tar(Y_new, Y, tar_agent, ego_agent) + Loss.ADE_loss_Y_GT_and_Y_pred(Y, Pred_t, tar_agent)

class HideCollisionLossPred(LossFunction):
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        return Loss.collision_loss_Y_ego_GT_and_Y_perturb_tar(Y_new, Y, tar_agent, ego_agent) + Loss.ADE_loss_Y_pred_and_Y_pred_iteration_1(Pred_t, Pred_iter_1)

# Specific barrier function implementations
class LogBarrier(BarrierFunction):
    def __init__(self, distance_threshold, log_value):
        self.distance_threshold = distance_threshold
        self.log_value = log_value

    def calculate_barrier(self, X_new, X, tar_agent):
        return Loss.barrier_log_function(self.distance_threshold, X_new, X, self.log_value, tar_agent)

class SplineBarrier(BarrierFunction):
    def __init__(self, distance_threshold, spline_data, log_value):
        self.distance_threshold = distance_threshold
        self.spline_data = spline_data
        self.log_value = log_value

    def calculate_barrier(self, X_new, X, tar_agent):
        return Loss.barrier_log_function_spline(self.distance_threshold, X_new, self.spline_data, self.log_value, tar_agent)