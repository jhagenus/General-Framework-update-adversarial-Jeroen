import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from matplotlib import gridspec
import torch

from Adversarial_classes.helper import Helper
from Adversarial_classes.spline import Spline
from Adversarial_classes.control_action import Control_action

class Plot:
    @staticmethod
    def plot_results(X, X_new_pert, Y, Y_new_pert, Pred_t, Pred_iter_1, loss_store, plot_loss, future_action,static_adv_scene,animated_adv_scene,car_length,car_width,mask_values_X,mask_values_Y,plot_smoothing,X_pert_smoothed,Pred_pert_smoothed,X_unpert_smoothed,Pred_unpert_smoothed,sigmas,smoothing_method,dt,flip_dimensions,epsilon_acc,epsilon_curv):
        # Plot the loss over the iterations
        if plot_loss:
            Plot.plot_loss_over_iterations(loss_store)

        # Plot the static adversarial scene
        if static_adv_scene:
            Plot.plot_static_adv_scene(X,X_new_pert,Y,Y_new_pert,Pred_t,Pred_iter_1,future_action)

        # Plot the animated adversarial scene  
        if animated_adv_scene:
            Plot.plot_animated_adv_scene(X,X_new_pert,Y,Y_new_pert,Pred_t,Pred_iter_1,future_action,car_length,car_width,mask_values_X,mask_values_Y,dt,flip_dimensions,epsilon_acc,epsilon_curv)

        # Plot the randomized smoothing
        if plot_smoothing:
            Plot.plot_smoothing(X,X_new_pert,Y,Y_new_pert,Pred_t,Pred_iter_1,future_action,sigmas,X_pert_smoothed,Pred_pert_smoothed,X_unpert_smoothed,Pred_unpert_smoothed,smoothing_method)

    @staticmethod
    def Plot_data(X, Y, spline_data, plot_input, plot_spline):
        # Return early if plotting is not requested
        if plot_input == False:
            return
        
        # Iterate over each example in the data
        for i in range(X.shape[0]):
            plt.figure(figsize=(18,12))
            # Plot the spline data
            if plot_spline:
                plt.plot(spline_data[i,:,0], spline_data[i,:,1], marker='o', color='m', label='Spline plot',markersize=4,alpha=0.2)

            # Plot the input data
            for j in range(X.shape[1]):
                # Plot the past and future positions of the target and ego agents
                if j != X.shape[1]-1:
                    Plot.draw_arrow(X[i,j,:], Y[i,j,:], plt, 'y', 3,'-','dashed', r'Observed target agent ($X_{tar}$)',r'Future target agent ($Y_{tar}$)', 1, 1)
                else:
                    Plot.draw_arrow(X[i,j,:], Y[i,j,:], plt, 'b', 3,'-','dashed', r'Observed ego agent ($X_{ego}$)',r'Future ego agent ($Y_{ego}$)', 1, 1)
            
            # Set the plot limits and road lines
            offset = 10
            min_value_x, max_value_x, min_value_y, max_value_y = Helper.find_limits_data(X, Y, i)

            # Plot the road lines
            Plot.plot_road_lines(min_value_x, max_value_x, min_value_y, max_value_y, offset,plt)

            # Set the plot limits
            plt.xlim(min_value_x - offset, max_value_x + offset)  
            plt.ylim(min_value_y - 2 * offset, max_value_y + 2 * offset)
            plt.axis('equal')
            plt.title(f'Example {i} of batch - Scene plot')
            plt.legend()
            plt.show()

    @staticmethod
    def plot_smoothing(X,X_new_pert,Y,Y_new_pert,Pred_t,Pred_iter_1,future_action,sigmas,X_pert_smoothed,Pred_pert_smoothed,X_unpert_smoothed,Pred_unpert_smoothed,smoothing_method):
        # Plot the randomized smoothing
        for i in range(X.shape[0]):
            # loop over the sigmas
            for j in range(len(sigmas)):
                fig = plt.figure(figsize = (18,12), dpi=1920/16)
                fig.suptitle(f'Example {j} of batch, sigma {sigmas[j]} - Randomized smoothing plot')

                if X_pert_smoothed is not None and X_unpert_smoothed is not None:
                    ax = fig.add_subplot(2,2,1)
                    ax1 = fig.add_subplot(2,2,2)
                    ax2 = fig.add_subplot(2,1,2)
                else:
                    ax = fig.add_subplot(2,1,1)
                    ax2 = fig.add_subplot(2,1,2)

                if X_pert_smoothed is not None and X_unpert_smoothed is not None:
                    style = 'unperturbed'
                    style1 = 'perturbed'
                    style2 = 'adv_smoothed'
                elif X_pert_smoothed is not None and X_unpert_smoothed is None:
                    style = 'perturbed'
                    style2 = 'adv_smoothed'
                else:
                    style = 'unperturbed'
                    style2 = 'adv_smoothed'

                # Plot 1: Plot the adversarial scene
                Plot.plot_data_with_adv(X, X_new_pert, Y, Y_new_pert, Pred_t, Pred_iter_1, future_action,ax,i,X_pert_smoothed,Pred_pert_smoothed,X_unpert_smoothed,Pred_unpert_smoothed,style,j,smoothing_method)
                    
                # Set the plot limits and road lines
                offset = 10
                min_value_x, max_value_x, min_value_y, max_value_y = Helper.find_limits_data(X, Y, i)

                # Plot the road lines
                Plot.plot_road_lines(min_value_x, max_value_x, min_value_y, max_value_y, offset,ax)

                # Set the plot limits
                ax.set_xlim(-20, 15)  
                ax.set_ylim(-10, 5)
                ax.set_aspect('equal')
                ax.set_title(f'Predictions {style} scene plot')

                if X_pert_smoothed is not None and X_unpert_smoothed is not None:
                    # Plot 2) Plot the smoothed predictions
                    Plot.plot_data_with_adv(X, X_new_pert, Y, Y_new_pert, Pred_t, Pred_iter_1, future_action,ax1,i,X_pert_smoothed,Pred_pert_smoothed,X_unpert_smoothed,Pred_unpert_smoothed,style1,j,smoothing_method)

                    # Plot the road lines
                    Plot.plot_road_lines(min_value_x, max_value_x, min_value_y, max_value_y, offset,ax1)

                    # Set the plot limits
                    ax1.set_xlim(-20, 15)  
                    ax1.set_ylim(-10, 5)
                    ax1.set_aspect('equal')
                    ax1.set_title(f'Predictions {style1} scene plot')

                # Plot 3) Plot the adversarial scene with the smoothed predictions
                Plot.plot_data_with_adv(X, X_new_pert, Y, Y_new_pert, Pred_t, Pred_iter_1, future_action,ax2,i,X_pert_smoothed,Pred_pert_smoothed,X_unpert_smoothed,Pred_unpert_smoothed,style2,j,smoothing_method)

                # Plot the road lines
                Plot.plot_road_lines(min_value_x, max_value_x, min_value_y, max_value_y, offset,ax2)

                # Set the plot limits
                ax2.set_xlim(-50, 15)  
                ax2.set_ylim(-10, 5)
                ax2.set_aspect('equal')
                ax2.set_title('Adversarial scene plot with smoothed prediction')
                ax2.legend(loc='lower left')
                plt.show()

    @staticmethod
    def plot_static_adv_scene(X,X_new_pert,Y,Y_new_pert,Pred_t,Pred_iter_1,future_action):
        # Plot the static adversarial scene
        for i in range(X.shape[0]):
            plt.figure(figsize=(18,12))
            
            # Plot the data
            Plot.plot_data_with_adv(X, X_new_pert, Y, Y_new_pert, Pred_t, Pred_iter_1, future_action,plt,i)
                
            # Set the plot limits and road lines
            offset = 10
            min_value_x, max_value_x, min_value_y, max_value_y = Helper.find_limits_data(X, Y, i)

            # Plot the road lines
            Plot.plot_road_lines(min_value_x, max_value_x, min_value_y, max_value_y, offset,plt)

            # Set the plot limits
            plt.xlim(min_value_x - offset, max_value_x + offset)  
            plt.ylim(min_value_y - 2 * offset, max_value_y + 2 * offset)
            plt.axis('equal')
            plt.title(f'Example {i} of batch - Adversarial scene plot')
            plt.legend()
            plt.show()

    @staticmethod
    def control_action_interpolated_points(interpolated_data,mask_values_X,flip_dimensions,dt):
        control_action_tensor = torch.tensor(np.expand_dims(np.expand_dims(np.array(interpolated_data),axis=0),axis=0)).to(device='cuda')
        control_action,_,_ = Control_action.Reversed_Dynamical_Model(control_action_tensor, mask_values_X, flip_dimensions, control_action_tensor,dt)
        return control_action.cpu().detach().numpy()
    
    def create_control_action_animation(data_adv_future,data_tar,data_adv,data_tar_pred,data_ego,mask_values_X,flip_dimensions,dt,num_interpolations,future_action,ax_control_acc,ax_control_curve):
        if future_action:
            # interpolated_data_list = [interpolated_data_tar_adv_future,interpolated_data_tar,interpolated_data_tar_adv,interpolated_data_tar_Pred,interpolated_data_ego]
            data_list = [data_adv_future,data_tar,data_adv,data_tar_pred,data_ego]
            colors = ['red','yellow','red','m','blue']
            alphas = [1,1,0.3,0.3,1]
            init_acc_curv = [0,0,0,0,0]
            x = [1,2,3,4,5]
        else:
            # interpolated_data_list = [interpolated_data_tar,interpolated_data_tar_adv,interpolated_data_tar_Pred,interpolated_data_ego]
            data_list = [data_tar,data_adv,data_tar_pred,data_ego]
            colors = ['red','yellow','m','blue']
            alphas = [1,1,0.3,1]
            init_acc_curv = [0,0,0,0]
            x = [1,2,3,4]

        control_action_list = []

        for data in data_list:
            control_action_list.append(Plot.control_action_interpolated_points(data,mask_values_X,flip_dimensions,dt))

        control_action_list = np.array(control_action_list)
        control_action_list = np.repeat(control_action_list[:,:,:,:-1,:],num_interpolations-1,axis=3)

        acc_bar = ax_control_acc.bar(x, init_acc_curv, color=colors)
        curv_bar = ax_control_curve.barh(x, init_acc_curv, color=colors)

        for bar_acc,bar_curv, alpha in zip(acc_bar,curv_bar, alphas):
            bar_acc.set_alpha(alpha)
            bar_curv.set_alpha(alpha)

        return acc_bar, curv_bar, control_action_list

    @staticmethod
    def plot_animated_adv_scene(X,X_new_pert,Y,Y_new_pert,Pred_t,Pred_iter_1,future_action,car_length,car_width,mask_values_X,mask_values_Y,dt,flip_dimensions,epsilon_acc,epsilon_curv):
        for i in range(X.shape[0]):
            num_interpolations = 5
            dt_new = dt / (num_interpolations - 1)
            interpolated_data_tar = []
            interpolated_data_tar_Pred = []
            interpolated_data_tar_adv = []
            interpolated_data_tar_adv_future = []
            interpolated_data_ego = []

            # Interpolate the data to smooth the animation
            for j in range(X.shape[1]):
                if j != X.shape[1]-1:
                    agent = 'target'
                    data_tar = np.concatenate((X[i,j,:,:],Y[i,j,:,:]),axis=0)
                    interpolated_data = Spline.interpolate_points(data_tar, num_interpolations,agent,mask_values_X[i],mask_values_Y[i])
                    interpolated_data_tar.append(interpolated_data)

                    agent = 'adv'
                    data_tar_pred = np.concatenate((X[i,j,:,:],Pred_iter_1[i,:,:]),axis=0)
                    interpolated_data = Spline.interpolate_points(data_tar_pred, num_interpolations,agent)
                    interpolated_data_tar_Pred.append(interpolated_data)

                    data_adv = np.concatenate((X_new_pert[i,j,:,:],Pred_t[i,:,:]),axis=0)
                    interpolated_data_adv = Spline.interpolate_points(data_adv, num_interpolations,agent)
                    interpolated_data_tar_adv.append(interpolated_data_adv)

                    if future_action:
                        data_adv_future = np.concatenate((X_new_pert[i,j,:,:],Y_new_pert[i,j,:,:]),axis=0)
                        interpolated_data_adv_future = Spline.interpolate_points(data_adv_future, num_interpolations,agent)
                        interpolated_data_tar_adv_future.append(interpolated_data_adv_future)

                else:
                    agent = 'ego'
                    data_ego = np.concatenate((X[i,j,:,:],Y[i,j,:,:]),axis=0)
                    interpolated_data = Spline.interpolate_points(data_ego, num_interpolations,agent)
                    interpolated_data_ego.append(interpolated_data)

            # initialize the plot
            fig = plt.figure(figsize = (18,12), dpi=1920/16)
            fig.suptitle(f'Example {i} of batch - Adversarial scene plot animated')

            # Create subplots
            gs = gridspec.GridSpec(3, 4, figure=fig)
            ax = fig.add_subplot(gs[1, :3])
            ax1 = fig.add_subplot(gs[1, 3])
            ax2 = fig.add_subplot(gs[2, :])
            ax_control_acc = fig.add_subplot(gs[0, :2])
            ax_control_curve = fig.add_subplot(gs[0, 2:])

            # Acceleration animation
            # Order of the bars
            acc_bar, curv_bar, control_action_list = Plot.create_control_action_animation(data_adv_future,data_tar,data_adv,data_tar_pred,data_ego,mask_values_X,flip_dimensions,dt,num_interpolations,future_action,ax_control_acc,ax_control_curve)

            ax_control_acc.set_ylim(-epsilon_acc, epsilon_acc)
            ax_control_acc.set_title('Control action: Acceleration')

            ax_control_curve.set_xlim(-epsilon_curv, epsilon_curv)
            ax_control_curve.set_title('Control action: Curvature')

            # initialize the cars
            rectangles_tar_pred = Plot.add_rectangles(ax, interpolated_data_tar_Pred, 'm', r'Targent-agent ($\hat{Y}_{ego}$)', car_length, car_width,alpha=0.5)
            rectangles_tar = Plot.add_rectangles(ax, interpolated_data_tar, 'yellow', r'Target-agent ($X_{tar}$ and $Y_{tar}$)', car_length, car_width,alpha=1)
            rectangles_ego = Plot.add_rectangles(ax, interpolated_data_ego, 'blue', r'Ego-agent ($X_{ego}$ and $Y_{ego}$)', car_length, car_width,alpha=1)

            if future_action:
                rectangles_tar_adv_future = Plot.add_rectangles(ax,interpolated_data_tar_adv, 'red', r'Adversarial target agent ($\tilde{X}_{tar}$ and $\tilde{Y}_{tar}$)', car_length, car_width,alpha=1)
                rectangles_tar_adv = Plot.add_rectangles(ax,interpolated_data_tar_adv, 'red', r'Adversarial prediction ($\hat{\tilde{Y}}_{tar}$)', car_length, car_width, alpha=0.3)
            else: 
                rectangles_tar_adv = Plot.add_rectangles(ax,interpolated_data_tar_adv, 'red', r'Adversarial prediction ($\tilde{X}_{tar}$ and $\hat{\tilde{Y}}_{tar}$)', car_length, car_width, alpha=1)
            
            # Function to update the animated plot
            def update(num):
                # Update the location of the car
                Plot.update_box_position(interpolated_data_tar_Pred,rectangles_tar_pred, car_length, car_width,num)
                Plot.update_box_position(interpolated_data_tar,rectangles_tar, car_length, car_width,num)
                Plot.update_box_position(interpolated_data_tar_adv,rectangles_tar_adv, car_length, car_width,num) 
                Plot.update_box_position(interpolated_data_ego,rectangles_ego, car_length, car_width,num)

                for i in range(control_action_list.shape[0]):
                    acc_bar[i].set_height(control_action_list[i,0,0,num,0])
                    curv_bar[i].set_width(-control_action_list[i,0,0,num,1])

                # acc_bar.set_height(control_action_list[:,0,0,num,0])
                # curv_bar.set_width(control_action_list[:,0,0,num,1])

                if future_action:
                    Plot.update_box_position(interpolated_data_tar_adv_future,rectangles_tar_adv_future, car_length, car_width,num)

                return 
            
            # Set the plot limits and road lines
            # min_value_x, max_value_x, min_value_y, max_value_y = self.find_limits_data(X, Y, i)

            # Plot the road lines
            offset = 10
            Plot.plot_road_lines(-100, 20, -30, 20, offset,ax)

            # Set the plot limits
            ax.set_xlim(-100, 10) 
            ax.set_ylim(-30, 10)
            ax.set_aspect('equal')
            ax.legend(loc='lower left')
            ax.set_title('Animation of the adversarial scene')
            
            ani = animation.FuncAnimation(fig, update, len(interpolated_data_tar[0])-1,
                                        interval=dt_new*1000, blit=False)
            
            # Plot the second Figure
            Plot.plot_data_with_adv(X, X_new_pert, Y, Y_new_pert, Pred_t, Pred_iter_1, future_action,ax1,i)

            # Plot the road lines
            Plot.plot_road_lines(2, 10, -2, 4, offset,ax1)

            # Set the plot limits
            ax1.set_aspect('equal')
            ax1.set_xlim(2, 10)  
            ax1.set_ylim(-2, 4)
            ax1.set_title('Zoomed adversarial scene plot')

            # Plot the third Figure
            Plot.plot_data_with_adv(X, X_new_pert, Y, Y_new_pert, Pred_t, Pred_iter_1, future_action,ax2,i)

            # Plot the rectangle for zoom
            ax2.add_patch(patches.Rectangle((2, -2), 8, 6, edgecolor='black', facecolor='none', linestyle='dashed', linewidth=1))

            # include pointer
            # Adding an arrow to point from figure to figure
            arrow = FancyArrowPatch((0.85, 0.40), (0.81, 0.60),
                                    transform=fig.transFigure,  
                                    mutation_scale=20,         
                                    lw=1,                       
                                    arrowstyle="-|>",           
                                    color='black')             

            fig.patches.extend([arrow])

            # Plot the road lines
            offset = 10
            Plot.plot_road_lines(-100, 20, -30, 20, offset,ax2)

            # Set the plot limits
            ax2.set_xlim(-80, 10) 
            ax2.set_ylim(-15, 5)
            ax2.legend(loc='lower left')
            ax2.set_aspect('equal')
            ax2.set_title('Adversarial scene static')

            ani.save(f'basic_animation_new-{np.random.rand(1)}.mp4')
                        
            plt.show()

    @staticmethod
    def plot_loss_over_iterations(loss_store):
        # Plot the loss over the iterations
        loss_store = np.array(loss_store)
        plt.figure(0)
        for i in range(loss_store.shape[1]):
            plt.plot(loss_store[:,i], marker='o', linestyle='-',label=f'Sample {i}')
        plt.title('Loss for samples')
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_data_with_adv(X, X_new_pert, Y, Y_new_pert, Pred_t, Pred_iter_1, future_action,figure_input,index,X_pert_smoothed=None,Pred_pert_smoothed=None,X_unpert_smoothed=None,Pred_unpert_smoothed=None,style=None,index_sigma=None,smoothing_method=None):
        for j in range(X.shape[1]):
            if j != X.shape[1]-1:
                # Plot target agent
                Plot.draw_arrow(X[index,j,:], Y[index,j,:], figure_input, 'y', 3,'-', 'dashed', r'Observed target agent ($X_{tar}$)',r'Future target agent ($Y_{tar}$)', 1, 1)
                
                # Plot the prediction on unperturbed target agent
                if style != 'perturbed':
                    Plot.draw_arrow(X[index,j,:], Pred_iter_1[index,:], figure_input, 'm', 3,'-', '-', None,r'Prediction target agent ($\hat{Y}_{tar}$)', 0, 0.5)

                # Plot pertubed history target agent and prediction
                if style != 'unperturbed':
                    Plot.draw_arrow(X_new_pert[index,j,:], Pred_t[index,:], figure_input, 'r', 3,'-', '-', r'Adversarial target agent ($\tilde{X}_{tar}$)',r'Adversarial prediction ($\hat{\tilde{Y}}_{tar}$)', 1, 0.5)
            
                # Plot future perturbed target agent
                if future_action and style != 'unperturbed' and style != 'perturbed':
                    Plot.draw_arrow(X_new_pert[index,j,:], Y_new_pert[index,j,:], figure_input, 'r', 3,'-', 'dashed', None,r'Adversarial target agent ($\tilde{Y}_{tar}$)', 0, 1)
            
            else:
                Plot.draw_arrow(X[index,j,:], Y[index,j,:], figure_input, 'b', 3,'-', 'dashed', r'Observed ego agent ($X_{ego}$)',r'Future ego agent ($Y_{ego}$)', 1, 1)

            if j == 0:
                if style == 'unperturbed':
                    if smoothing_method == 'positio':
                        for k in range(Pred_unpert_smoothed.shape[1]):
                            Plot.draw_arrow(X[index,0,:], Pred_unpert_smoothed[index_sigma,k,index,:], figure_input, 'c', 3,'-','-', None,None, 0, 0.4)
                        
                        Average_Pred_unpert_smoothed = np.mean(Pred_unpert_smoothed,axis=1)
                        Plot.draw_arrow(X[index,0,:], Average_Pred_unpert_smoothed[index_sigma,index,:], figure_input, 'c', 3,'-','-', None,None, 0, 1)
                    else:
                        for k in range(Pred_unpert_smoothed.shape[1]):
                            Plot.draw_arrow(X_unpert_smoothed[index_sigma,k,index,0,:], Pred_unpert_smoothed[index_sigma,k,index,:], figure_input, 'c', 3,'-.','-', None,None, 0.4, 0.4)

                        average_X_unpert_smoothed = np.mean(X_unpert_smoothed,axis=1)
                        Average_Pred_unpert_smoothed = np.mean(Pred_unpert_smoothed,axis=1)
                        Plot.draw_arrow(average_X_unpert_smoothed[index_sigma,index,0,:], Average_Pred_unpert_smoothed[index_sigma,index,:], figure_input, 'c', 3,'-.','-', None,None, 1, 1)

                elif style == 'perturbed':
                    if smoothing_method == 'position':
                        for k in range(Pred_pert_smoothed.shape[1]):
                            Plot.draw_arrow(X_new_pert[index,0,:], Pred_pert_smoothed[index_sigma,k,index,:], figure_input, 'g', 3,'-','-', None,None, 0, 0.4)
                        
                        Average_Pred_pert_smoothed = np.mean(Pred_pert_smoothed,axis=1)
                        Plot.draw_arrow(X_new_pert[index,0,:], Average_Pred_pert_smoothed[index_sigma,index,:], figure_input, 'g', 3,'-','-', None,None, 0, 1)
                    else:
                        for k in range(Pred_pert_smoothed.shape[1]):
                            Plot.draw_arrow(X_pert_smoothed[index_sigma,k,index,0,:], Pred_pert_smoothed[index_sigma,k,index,:], figure_input, 'g', 3,'-.','-', None,None, 0.4, 0.4)

                        average_X_pert_smoothed = np.mean(X_pert_smoothed,axis=1)
                        Average_Pred_pert_smoothed = np.mean(Pred_pert_smoothed,axis=1)
                        Plot.draw_arrow(average_X_pert_smoothed[index_sigma,index,0,:], Average_Pred_pert_smoothed[index_sigma,index,:], figure_input, 'g', 3,'-.','-', None,None, 1, 1)

                elif style == 'adv_smoothed':
                    if smoothing_method == 'position':
                        if Pred_unpert_smoothed is not None and Pred_pert_smoothed is None:
                            Average_Pred_unpert_smoothed = np.mean(Pred_unpert_smoothed,axis=1)
                            Plot.draw_arrow(X_new_pert[index,0,:], Average_Pred_unpert_smoothed[index_sigma,index,:], figure_input, 'c', 3,'-','-', None,r'Smoothed target agent ($\bar{X}_{tar}$)', 0, 1)
                        
                        elif Pred_unpert_smoothed is None and Pred_pert_smoothed is not None:
                            Average_Pred_pert_smoothed = np.mean(Pred_pert_smoothed,axis=1)
                            Plot.draw_arrow(X_new_pert[index,0,:], Average_Pred_pert_smoothed[index_sigma,index,:], figure_input, 'g', 3,'-','-', None,r'Smoothed adversarial target agent ($\bar{\tilde{X}}_{tar}$)', 0, 1)
                        else:
                            Average_Pred_unpert_smoothed = np.mean(Pred_unpert_smoothed,axis=1)
                            Plot.draw_arrow(X[index,0,:], Average_Pred_unpert_smoothed[index_sigma,index,:], figure_input, 'c', 3,'-','-', None,r'Smoothed target agent ($\bar{X}_{tar}$)', 0, 1)

                            Average_Pred_pert_smoothed = np.mean(Pred_pert_smoothed,axis=1)
                            Plot.draw_arrow(X_new_pert[index,0,:], Average_Pred_pert_smoothed[index_sigma,index,:], figure_input, 'g', 3,'-','-', None,r'Smoothed adversarial target agent ($\bar{\tilde{X}}_{tar}$)', 0, 1)
                    else:
                        if Pred_unpert_smoothed is not None and Pred_pert_smoothed is None:
                            average_X_unpert_smoothed = np.mean(X_unpert_smoothed,axis=1)
                            Average_Pred_unpert_smoothed = np.mean(Pred_unpert_smoothed,axis=1)
                            Plot.draw_arrow(average_X_unpert_smoothed[index_sigma,index,0,:], Average_Pred_unpert_smoothed[index_sigma,index,:], figure_input, 'c', 3,'-.','-', r'Smoothed target agent ($\bar{X}_{tar}$)',r'Smoothed prediction target agent ($\hat{\bar{Y}}_{tar}$)', 1, 0.5)
                        
                        elif Pred_unpert_smoothed is None and Pred_pert_smoothed is not None:
                            average_X_pert_smoothed = np.mean(X_pert_smoothed,axis=1)
                            Average_Pred_pert_smoothed = np.mean(Pred_pert_smoothed,axis=1)
                            Plot.draw_arrow(average_X_pert_smoothed[index_sigma,index,0,:], Average_Pred_pert_smoothed[index_sigma,index,:], figure_input, 'g', 3,'-.','-', r'Smoothed adversarial target agent ($\bar{\tilde{X}}_{tar}$)',r'Smoothed adversarial prediction target agent ($\hat{\bar{\tilde{Y}}}_{tar}$)', 1, 0.5)
                        else:
                            average_X_unpert_smoothed = np.mean(X_unpert_smoothed,axis=1)
                            Average_Pred_unpert_smoothed = np.mean(Pred_unpert_smoothed,axis=1)
                            Plot.draw_arrow(average_X_unpert_smoothed[index_sigma,index,0,:], Average_Pred_unpert_smoothed[index_sigma,index,:], figure_input, 'c', 3,'-.','-', r'Smoothed target agent ($\bar{X}_{tar}$)',r'Smoothed prediction target agent ($\hat{\bar{Y}}_{tar}$)', 1, 0.5)

                            average_X_pert_smoothed = np.mean(X_pert_smoothed,axis=1)
                            Average_Pred_pert_smoothed = np.mean(Pred_pert_smoothed,axis=1)
                            Plot.draw_arrow(average_X_pert_smoothed[index_sigma,index,0,:], Average_Pred_pert_smoothed[index_sigma,index,:], figure_input, 'g', 3,'-.','-', r'Smoothed adversarial target agent ($\bar{\tilde{X}}_{tar}$)',r'Smoothed adversarial prediction target agent ($\hat{\bar{\tilde{Y}}}_{tar}$)', 1, 0.5)


    @staticmethod
    def draw_arrow(data_X, data_Y, figure_input, color, linewidth,line_style_input,line_style_output, label_input,label_output,alpha_input,alpha_output):
        figure_input.plot(data_X[:,0], data_X[:,1], linestyle=line_style_input,linewidth=linewidth, color=color, label=label_input,alpha=alpha_input)
        figure_input.plot((data_X[-1,0],data_Y[0,0]), (data_X[-1,1],data_Y[0,1]), linestyle=line_style_output,linewidth=linewidth, color=color,alpha=alpha_output)
        figure_input.plot(data_Y[:-1,0], data_Y[:-1,1], linestyle=line_style_output,linewidth=linewidth, color=color,alpha=alpha_output,label=label_output)
        figure_input.annotate('', xy=(data_Y[-1,0], data_Y[-1,1]), xytext=(data_Y[-2,0], data_Y[-2,1]),
                size=20,arrowprops=dict(arrowstyle='-|>',linestyle=None,color=color,lw=linewidth,alpha=alpha_output))

    @staticmethod
    def add_rectangles(figure_input, data_list, color, label, car_length, car_width,alpha=1):
        rectangles = []
        # Add rectangles to the plot
        for _ in range(len(data_list)):
            rect = patches.Rectangle((0,0), car_length, car_width, edgecolor='none', facecolor=color, label=label,alpha=alpha)
            figure_input.add_patch(rect)
            rectangles.append(rect)
        # To only add one label per type in the legend
        handles, labels = figure_input.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        figure_input.legend(by_label.values(), by_label.keys())

        return rectangles
    
    @staticmethod
    def update_box_position(data,rectangle_data, car_length, car_width,num):
        # Compensate that the rectangle is drawn from the bottom left corner
        for i in range(len(rectangle_data)):
            x, y = data[i][:,0], data[i][:,1]
            dx = x[num + 1] - x[num]
            dy = y[num + 1] - y[num]
            angle_rad = np.arctan2(dy, dx)
            shift_x = (car_width / 2) * np.sin(angle_rad) - (car_length / 2) * np.cos(angle_rad)
            shift_y = -(car_width / 2) * np.cos(angle_rad) - (car_length / 2) * np.sin(angle_rad)
            rectangle_data[i].set_xy([x[num-1] + shift_x, y[num-1] + shift_y])
            angle = np.arctan2(dy, dx) * (180 / np.pi)  
            rectangle_data[i].set_angle(angle)

    @staticmethod
    def plot_road_lines(min_value_x, max_value_x, min_value_y, max_value_y, offset,figure_input):
        # Plot the dashed road lines
        y_dash = [0,0]
        x_min_dash = [min_value_x - offset, 4.5]
        x_max_dash = [-4.5, max_value_x + offset]

        x_dash = [0,0]
        y_min_dash = [min_value_y - 2 * offset, 4.5]
        y_max_dash = [-4.5, max_value_y + 2 * offset]

        figure_input.hlines(y_dash,x_min_dash,x_max_dash, linestyle='dashed', colors='k',linewidth=0.75)
        figure_input.vlines(x_dash,y_min_dash,y_max_dash, linestyle='dashed', colors='k',linewidth=0.75)
        
        # Plot the solid road lines
        y_solid = [-3.5, -3.5, 3.5, 3.5]
        x_min_solid = [min_value_x - offset, 3.5, min_value_x - offset, 3.5]
        x_max_solid = [-3.5, max_value_x + offset, -3.5, max_value_x + offset]

        x_solid = [-3.5, 3.5, 3.5, -3.5]
        y_min_solid = [min_value_y - 2 * offset, min_value_y - 2 * offset, 3.5, 3.5]
        y_max_solid = [-3.5, -3.5, max_value_y + 2 * offset, max_value_y + 2 * offset]

        figure_input.hlines(y_solid,x_min_solid,x_max_solid, linestyle="solid", colors='k')
        figure_input.vlines(x_solid,y_min_solid,y_max_solid, linestyle="solid", colors='k')
        

    