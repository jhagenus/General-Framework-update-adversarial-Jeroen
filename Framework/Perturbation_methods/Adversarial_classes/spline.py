import numpy as np
from scipy.interpolate import CubicSpline

from Adversarial_classes.helper import Helper

class Spline:
    @staticmethod
    def spline_data(X, Y, mask_values_X, mask_values_Y, flip_dimensions, spline_interval, spline):
        if spline == False:
            return None
        
        # Combine historical and future data
        # JULIAN: Using axis = -2 is safer, as there might be additional axes previously (e.g., for 20 predictions)
        Spline_input_values = np.concatenate((X,Y),axis=2)
        
        # Check edge case scenarios where agent is standing 
        # JULIAN: This whole thing right now is very dangerous, as it is only valid in this certain scenario
        # JULIAN: try to rewrite it to be also applicable for different scenarios
        # JULIAN: Additionally, I am not sure if monotony is actually necessary here
        if flip_dimensions:
            # JULIAN: This could easily be vectorized
            for batch_idx in range(X.shape[0]):
                # Check if the target agent is standing still in historical data
                if mask_values_X[batch_idx] == True:
                    # check if the agent moves in the future data -> store the first and last point of historical data -> store all future data -> set the rest to nan
                    if mask_values_Y[batch_idx] == False:
                            Spline_input_values[batch_idx,0,1,:] = Spline_input_values[batch_idx,0,X.shape[2],:]
                            Spline_input_values[batch_idx,0,2:2+Y.shape[2],:] = Spline_input_values[batch_idx,0,X.shape[2]:,:]
                            Spline_input_values[batch_idx, 0, 2+Y.shape[2]:,:] = np.nan

                    # check if the agent is standing still in the future data -> store the lastest future point -> set the rest to nan (bug in data recording)
                    else:
                        for i in reversed(range(Spline_input_values.shape[2])):
                            # Check if the new positions in the future data are smaller than the last position in the historical data
                            if Spline_input_values[batch_idx, 0, i, 0] < Spline_input_values[batch_idx, 0, 0, 0]:
                                Spline_input_values[batch_idx,0,1,:] = Spline_input_values[batch_idx,0,i,:]
                                Spline_input_values[batch_idx, 0, 2:,:] = np.nan
                                break
                
                # Remove values that are not monotonic For the spline function
                i = 1
                while i < Spline_input_values.shape[2]:
                    if Spline_input_values[batch_idx, 0, i, 0] > Spline_input_values[batch_idx, 0, i - 1, 0]:
                        Spline_input_values[batch_idx, 0, i-1:,:] = np.nan
                        break
                    else:
                        i += 1

        # Initialize spline data
        spline_data = np.zeros((X.shape[0],spline_interval,2))

        # Spline historical data
        for i in range(X.shape[0]):
            # Extract the spline data
            sample_spline = Spline_input_values[i,0,:,:]

            # Remove NaN values
            sample_spline = sample_spline[~np.isnan(sample_spline).any(axis=1)]

            # Flip data if it is not in the correct order for cubic spline funciton
            if sample_spline[0,0] > sample_spline[-1,0]:
                sample_spline[:,0] = np.flip(sample_spline[:,0])
            if sample_spline[0,1] > sample_spline[-1,1]:
                sample_spline[:,1] = np.flip(sample_spline[:,1])

            # Sample from cubic spline function
            Spline_function = CubicSpline(sample_spline[:,0], sample_spline[:,1])
            xs = np.linspace(sample_spline[0,0], sample_spline[-1,0], spline_interval)
            spline_data[i,:,0] = xs
            spline_data[i,:,1] = Spline_function(xs)

        return spline_data
    
    @staticmethod
    def interpolate_points(data,num_interpolations,agent,mask_values_X = False,mask_values_Y = False):
        # Flip agent to make x values monotonic

        # JULIAN: I think that this whole aspect is far to complicated. Especially with the generation of the interpolated points
        # JULIAN: One can analittically calculate the shortest distance of a point to a line between two other points.
        # JULIAN: Simply do this for all the line segemnts in the trajectory, and then choose the minimum distance
        # JULIAN: Doing this will likely be much faster because of more efficient vectorization as well.
        # JULIAN: See first answer at https://math.stackexchange.com/questions/330269/the-distance-from-a-point-to-a-line-segment
        monotonic = Helper.is_monotonic(data)
        if agent == 'target':
            if mask_values_X or mask_values_Y:
                if monotonic:
                    new_data = np.flip(data, axis=0)
                else:
                    new_data = np.flip(np.flip(data, axis=1),axis=0)
                # add offset because some valus are the same
                for i in range(1,new_data.shape[0]):
                    new_data[i,0] += 0.0001*i
            else:
                new_data = np.flip(np.flip(data, axis=1),axis=0)
            spline = CubicSpline(new_data[:, 0], new_data[:, 1])
        elif agent == 'adv':
            if monotonic:
                new_data = np.flip(data, axis=0)
            else:
                new_data = np.flip(np.flip(data, axis=1),axis=0)
            spline = CubicSpline(new_data[:, 0], new_data[:, 1])
        else:
            new_data = data
            spline = CubicSpline(new_data[:, 0], new_data[:, 1])

        interpolated_points = []

        # interpolate the data
        for i in range(len(data[:, 0]) - 1):
            x_interval = np.linspace(new_data[i, 0], new_data[i+1, 0], num_interpolations)
            y_interval = spline(x_interval)
            if i == len(data[:, 0]) - 1:
                interpolated_points.extend(zip(x_interval, y_interval))
            else:
                interpolated_points.extend(zip(x_interval[:-1], y_interval[:-1]))
        
        interpolated_points = np.array(interpolated_points)

        if agent == 'target':
            if mask_values_X or mask_values_Y:
                interpolated_points = np.flip(interpolated_points, axis=0)
            else:
                interpolated_points = np.flip(np.flip(interpolated_points, axis=1),axis=0)
        elif agent == 'adv':
            if monotonic:
                interpolated_points = np.flip(interpolated_points, axis=0)
            else:
                interpolated_points = np.flip(np.flip(interpolated_points, axis=1),axis=0)
        return interpolated_points