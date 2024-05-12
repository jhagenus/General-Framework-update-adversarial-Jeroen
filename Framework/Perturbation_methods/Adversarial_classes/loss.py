import torch

class Loss:
    @staticmethod
    def calculate_loss(X,X_new,Y,Y_new,Pred_t,Pred_iter_1,ADE_loss,ADE_loss_adv_future_GT,ADE_loss_adv_future_pred,collision_loss,fake_collision_loss_GT,fake_collision_loss_Pred,hide_collision_loss_GT,hide_collision_loss_Pred,log_barrier,ADVDO_barrier,spline_barrier,distance_threshold,log_value,spline_data):

        # Add  regularization loss to adversarial input using barrier function
        if log_barrier:
            barrier_output = Loss.barrier_log_function(distance_threshold, X_new, X, log_value)
            barrier_output_check = True
        elif ADVDO_barrier:
            barrier_output = Loss.AVDDO_barrier_function(X_new, X, distance_threshold)
            barrier_output_check = True
        elif spline_barrier:
            barrier_output = Loss.barrier_log_function_spline(distance_threshold, X_new, spline_data, log_value)
            barrier_output_check = True
        else:
            barrier_output_check = False
            
        # Calculate the total loss
        if ADE_loss:
            if not barrier_output_check:
                losses = -Loss.ADE_loss_function(Y, Pred_t)
            else:
                losses = -Loss.ADE_loss_function(Y, Pred_t) - barrier_output
        elif ADE_loss_adv_future_GT:
            if not barrier_output_check:
                losses = -Loss.ADE_adv_future_pred(Y_new, Pred_t) + Loss.ADE_adv_future_GT(Y_new, Y)
            else:
                losses = -Loss.ADE_adv_future_pred(Y_new, Pred_t) + Loss.ADE_adv_future_GT(Y_new, Y) - barrier_output
        elif ADE_loss_adv_future_pred:
            if not barrier_output_check:
                losses = -Loss.ADE_adv_future_pred(Y_new, Pred_t) + Loss.ADE_adv_future_pred_iter_1(Y_new, Pred_iter_1)
            else:
                losses = -Loss.ADE_adv_future_pred(Y_new, Pred_t) + Loss.ADE_adv_future_pred_iter_1(Y_new, Pred_iter_1) - barrier_output
        elif collision_loss:
            if not barrier_output_check:
                losses = Loss.collision_loss_function(Y, Pred_t)
            else:
                losses = Loss.collision_loss_function(Y, Pred_t) - barrier_output
        elif fake_collision_loss_GT:
            if not barrier_output_check:
                losses = Loss.collision_loss_function(Y, Pred_t) + Loss.ADE_adv_future_GT(Y_new, Y)
            else:
                losses = Loss.collision_loss_function(Y, Pred_t) + Loss.ADE_adv_future_GT(Y_new, Y) - barrier_output 
        elif fake_collision_loss_Pred:
            if not barrier_output_check:
                losses = Loss.collision_loss_function(Y, Pred_t) + Loss.ADE_adv_future_pred_iter_1(Y_new, Pred_iter_1)
            else:
                losses = Loss.collision_loss_function(Y, Pred_t) + Loss.ADE_adv_future_pred_iter_1(Y_new, Pred_iter_1) - barrier_output
        elif hide_collision_loss_GT:
            if not barrier_output_check:
                losses = Loss.collision_loss_adv_future(Y_new, Y) + Loss.ADE_loss_function(Y, Pred_t) 
            else:
                losses = Loss.collision_loss_adv_future(Y_new, Y) + Loss.ADE_loss_function(Y, Pred_t) - barrier_output
        elif hide_collision_loss_Pred:
            if not barrier_output_check:
                losses = Loss.collision_loss_adv_future(Y_new, Y) + Loss.ADE_pred_pred_iter_1(Pred_t, Pred_iter_1)
            else:
                losses = Loss.collision_loss_adv_future(Y_new, Y) + Loss.ADE_pred_pred_iter_1(Pred_t, Pred_iter_1) - barrier_output

        return losses
    
    @staticmethod
    def ADE_loss_function(Y, Pred_t):
        # index 0 is the target agent -> norm is over the positions -> inner torch.mean is over the time steps -> outer torch.mean is over the number of prediciton
        return torch.mean(torch.mean(torch.linalg.norm(Y[:,0,:,:].unsqueeze(1) - Pred_t[:,:,:,:], dim=-1 , ord = 2), dim=-1),dim=-1) 
    
    @staticmethod
    def ADE_adv_future_pred(Y_new, Pred_t):
        # index 0 is the target agent -> norm is over the positions -> inner torch.mean is over the time steps -> outer torch.mean is over the number of prediciton
        return torch.mean(torch.mean(torch.linalg.norm(Y_new[:,0,:,:].unsqueeze(1) - Pred_t[:,:,:,:], dim=-1 , ord = 2), dim=-1),dim=-1) 
    
    def ADE_adv_future_pred_iter_1(Y_new, Pred_iter_1):
        # index 0 is the target agent -> norm is over the positions -> inner torch.mean is over the time steps -> outer torch.mean is over the number of prediciton
        return torch.mean(torch.mean(torch.linalg.norm(Y_new[:,0,:,:].unsqueeze(1) - Pred_iter_1[:,:,:,:], dim=-1 , ord = 2), dim=-1),dim=-1)
    
    def ADE_pred_pred_iter_1(Pred_t, Pred_iter_1):
        # The target agent -> norm is over the positions -> inner torch.mean is over the time steps -> outer torch.mean is over the number of prediciton
        return torch.mean(torch.mean(torch.linalg.norm(Pred_t[:,:,:,:] - Pred_iter_1[:,:,:,:], dim=-1 , ord = 2), dim=-1),dim=-1)
    
    @staticmethod
    def ADE_adv_future_GT(Y_new, Y):
        # index 0 is the target agent -> norm is over the positions -> torch.mean is over the time steps 
        return torch.mean(torch.linalg.norm(Y_new[:,0,:,:] - Y[:,0,:,:], dim=-1 , ord = 2), dim=-1) 
    
    @staticmethod
    def collision_loss_function(Y, Pred_t):
        # index 1 is the ego agent -> norm is over the positions -> torch.min is over the time steps -> torch.mean is over the number of prediciton
        return torch.mean(torch.linalg.norm(Y[:,1,:,:].unsqueeze(1) - Pred_t[:,:,:,:], dim=-1 , ord = 2).min(dim=-1).values,dim=-1)
    
    @staticmethod
    def collision_loss_adv_future(Y_new, Y):
        # index 0 is target agent, index 1 is the ego agent -> norm is over the positions -> torch.min is over the time steps 
        return torch.linalg.norm(Y[:,1,:,:] - Y_new[:,0,:,:], dim=-1 , ord = 2).min(dim=-1).values

    @staticmethod
    def barrier_log_function(distance_threshold, X_new, X, log_value):
        # index 0 is the target agent -> norm is over the positions
        barrier_norm = torch.norm(X_new[:,0,:,:] - X[:,0,:,:], dim = -1)
        # log barrier function
        barrier_log = torch.log(distance_threshold - barrier_norm)
        # normalize the log barrier function
        barrier_log_new = barrier_log / torch.log(torch.tensor(log_value))
        # mean over the time steps
        barrier_output = torch.mean(barrier_log_new,dim=-1)
        return barrier_output
    
    @staticmethod
    def AVDDO_barrier_function(X_new, X, distance_threshold_past):
        # Other paper less relevant
        barrier_norm = torch.norm(X_new[:,0,:,:] - X[:,0,:,:], dim = -1)
        barrier_output = -((barrier_norm / distance_threshold_past) - torch.sigmoid(barrier_norm / distance_threshold_past) + 0.5).sum(dim=-1)
        return barrier_output
    
    @staticmethod
    def barrier_log_function_spline(distance_threshold_past, X_new, spline_data, spline_value_past):
        # index 0 is the target agent -> compare all points of agent with spline
        distance = torch.cdist(X_new[:,0,:,:], spline_data, p=2)
        # find the closest point on the spline
        min_indices = torch.argmin(distance, dim=-1)
        closest_points = torch.gather(spline_data.unsqueeze(1).expand(-1, X_new.shape[2], -1, -1), 2, min_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, spline_data.size(-1)))
        closest_points = closest_points.squeeze(2)
        # norm is over the positions
        barrier_norm = torch.norm(X_new[:,0,:,:] - closest_points, dim = -1)
        # log barrier function
        barrier_output = torch.log(distance_threshold_past - barrier_norm)
        # normalize the log barrier function
        barrier_log_new = barrier_output / torch.log(torch.tensor(spline_value_past))
        # mean over the time steps
        barrier_output = torch.mean(barrier_log_new,dim=-1)
        return barrier_output