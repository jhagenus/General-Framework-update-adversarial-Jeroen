import numpy as np
from scipy.interpolate import CubicSpline

from Adversarial_classes.helper import Helper

class Spline:
    @staticmethod
    def spline_data(X, Y, total_spline_values):
        # concatenate the data
        data = np.concatenate((X,Y),axis=-2)

        # check if data monotonic _> needed for spline function -> flip X,Y values if not
        monotonic_data = Helper.is_monotonic(data)
        data[~monotonic_data, :, :] = np.flip(data[~monotonic_data, :, :], axis=-1)

        # check if values are increasing -> needed for spline function -> flip trajectory if not
        increasing_data = Helper.is_increasing(data)
        data[~increasing_data, :, :] = np.flip(data[~increasing_data, :, :], axis=-2)

        # Cubic spline data
        spline_data = np.empty((X.shape[0], X.shape[1], total_spline_values, X.shape[3]))

        # Spline all the data
        for i in range(spline_data.shape[0]):
            for j in range(spline_data.shape[1]):
                x = np.linspace(data[i, j, 0, 0], data[i, j, -1, 0], total_spline_values)
                cs = CubicSpline(data[i, j, :, 0], data[i, j, :, 1])
                spline_data[i, j, :, 0] = x
                spline_data[i, j, :,1] = cs(x)

        # Translate back to original data
        spline_data[~increasing_data, :, :] = np.flip(spline_data[~increasing_data, :, :], axis=-2)
        spline_data[~monotonic_data, :, :] = np.flip(spline_data[~monotonic_data, :, :], axis=-1)

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