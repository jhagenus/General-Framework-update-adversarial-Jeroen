#%%
import copy
from experiment import Experiment

# Draw latex figure
Experiment_name = 'No Perturbations'
new_experiment = Experiment(Experiment_name)


#%% Select modules


# Select the models to be trained
Models = [{'model': 'trajectron_salzmann_old','kwargs': {'seed': 0, 'predict_ego': False}}]

# Select the params for the datasets to be considered
Data_params = [{'dt': 0.1, 'num_timesteps_in': (12,12), 'num_timesteps_out': (12, 12)}] 

# Select the datasets
Data_sets = [{'scenario': 'CoR_left_turns',  'max_num_agents': None, 't0_type': 'col_set', 'conforming_t0_types': []}]


# Select the spitting methods to be considered
Splitters = [{'Type': 'no_split', 'repetition': 0, 'train_pert': False, 'test_pert': False}]

# Select the metrics to be used
Metrics = ['ADE20_indep', 'FDE20_indep', 'Collision_rate_indep', 
           'Past_Acceleration_indep', 'Past_Curvature_indep']


new_experiment.set_modules(Data_sets, Data_params, Splitters, Models, Metrics)

#%% Other settings
# Set the number of different trajectories to be predicted by trajectory prediction models.
num_samples_path_pred = 20

# Deciding wether to enforce start times (or allow later predictions if not enough input data is available)
enforce_prediction_times = True

# determine if the upper bound for n_O should be enforced, or if prediction can be made without
# underlying output data (might cause training problems)
enforce_num_timesteps_out = True

# Determine if the useless prediction (i.e, prediction you cannot act anymore)
# should be exclude from the dataset
exclude_post_crit = False

# Decide wether missing position in trajectory data can be extrapolated
allow_extrapolation = True

# Use all available agents for predictions
agents_to_predict = 'predefined'

# Determine if allready existing results shoul dbe overwritten, or if not, be used instead
overwrite_results = 'no'

# Determine if the model should be evaluated on the training set as well
evaluate_on_train_set = False

# Determine if predictions should be saved
save_predictions = True

# Select method used for transformation function to path predictions
model_for_path_transform = 'trajectron_salzmann_old'

new_experiment.set_parameters(model_for_path_transform, num_samples_path_pred, 
                              enforce_num_timesteps_out, enforce_prediction_times, 
                              exclude_post_crit, allow_extrapolation, 
                              agents_to_predict, overwrite_results, 
                              save_predictions, evaluate_on_train_set)

#%% Run experiment
new_experiment.run()

# Load results
Results = new_experiment.load_results()
