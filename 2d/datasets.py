import numpy as np
from sklearn.model_selection import train_test_split
from data.dataset_generator import double_sde


class BaseDataset:
    def __init__(self, env_name=None, val_split=None,config=None):
        self.env_name = env_name
        self.val_split = val_split
        self.config = config
        self.train_trajs, self.val_trajs, self.state_dim = self.load_dataset()
        self.print_dataset_stats(self.train_trajs)
    
    def load_dataset(self):
        raise NotImplementedError("Subclasses must implement this method")

    def get_train_data(self):
        return self.train_trajs

    def get_val_data(self):
        return self.val_trajs
    
    def get_state_dim(self):
        return self.state_dim

    def print_dataset_stats(self, dataset):
        print("Number of trajectories:", len(dataset))
        
        sample_traj = dataset[0]
        
        if isinstance(sample_traj, dict):
            print("Observation shape:", sample_traj['obs'].shape)
            print("Action shape:", sample_traj['action'].shape)
            print("Observation - Min:", np.min(sample_traj['obs']), "Max:", np.max(sample_traj['obs']))
            print("Action - Min:", np.min(sample_traj['action']), "Max:", np.max(sample_traj['action']))
        else:
            print("Data shape:", sample_traj.shape)
            print("Data - Min:", np.min(sample_traj), "Max:", np.max(sample_traj))

    def normalize(self, data):
        raise NotImplementedError("Subclasses must implement this method")

    def unnormalize(self, data):
        raise NotImplementedError("Subclasses must implement this method")


class BranchingDataset(BaseDataset):
    def __init__(self, env_name='toy', val_split=0.1,config=None):
        self.obs_horizon = config["dataset"]["obs_horizon"]
        super().__init__(env_name, val_split,config)
    def load_dataset(self):

        trajectories = double_sde(g_constant=0.0)

        # Convert videos to trajectories
        trajs = []
        data_shape = None

        # Process training trajectories
        for trajectory in trajectories:
            data_shape = trajectory.shape  # (length, dim)
            len = trajectory.shape[0]

            traj = []
            for j in range(len - self.obs_horizon + 1):
                points = [trajectory[j+k] for k in range(self.obs_horizon)]
                point = np.stack(points, axis=0).flatten()
                traj.append(point)

            if traj:  # Only append if trajectory is not empty
                trajs.append(np.array(traj))


        if data_shape is None:
            raise ValueError("No valid trajectories were created")

        data_shape = (self.obs_horizon, *data_shape[1:])

        train_trajs, val_trajs = train_test_split(trajs, test_size=self.val_split, random_state=42)
        return train_trajs, val_trajs, data_shape
    


def get_dataset(env_name, **kwargs):
    if env_name == "base":
        return BaseDataset(env_name, **kwargs)
    elif env_name == "branching":
        return BranchingDataset(env_name, **kwargs)