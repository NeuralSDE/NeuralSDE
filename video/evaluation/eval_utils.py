import h5py
import torch

def save_tensors_to_h5(tensor_dict, path):
    """
    Save a dictionary of tensors to HDF5 format
    Args:
        tensor_dict: Dictionary containing tensors
        path: Path to save the HDF5 file (e.g., 'data.h5')
    """
    with h5py.File(path, 'w') as f:
        for key, tensor in tensor_dict.items():
            # Handle scalar tensors differently
            if isinstance(tensor, torch.Tensor) and tensor.ndim == 0:
                # Convert scalar tensor to a single value
                f.create_dataset(key, data=tensor.item())
            else:
                # Handle regular tensors as before
                f.create_dataset(key, 
                    data=tensor.detach().cpu().numpy(),
                    compression="gzip",
                    compression_opts=9)

def load_tensors_from_h5(filepath, device='cuda'):
    """
    Load tensors from HDF5 file
    Args:
        filepath: Path to the HDF5 file
        device: Target device for loaded tensors (default: 'cuda')
    Returns:
        Dictionary containing loaded tensors
    """
    tensor_dict = {}
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            data = f[key][()]  # Use [()] instead of [:] for scalar compatibility
            if isinstance(data, (float, int)):
                # Handle scalar values
                tensor_dict[key] = torch.tensor(data).to(device)
            else:
                # Handle regular arrays
                tensor_dict[key] = torch.from_numpy(data).to(device)
    return tensor_dict

def save_tensors_chunked(tensor_dict, filepath, chunk_size=32):
    """
    Save large tensor dictionary with chunking for better memory efficiency
    Args:
        tensor_dict: Dictionary containing tensors
        filepath: Path to save the HDF5 file
        chunk_size: Size of chunks for storage (default: 32)
    """
    with h5py.File(filepath, 'w') as f:
        for key, tensor in tensor_dict.items():
            # Handle scalar tensors differently
            if isinstance(tensor, torch.Tensor) and tensor.ndim == 0:
                # Convert scalar tensor to a single value
                f.create_dataset(key, data=tensor.item())
            else:
                tensor_np = tensor.cpu().numpy()
                # Define chunk size: (chunk_size, *original_dimensions)
                chunks = (min(chunk_size, tensor_np.shape[0]),) + tensor_np.shape[1:]
                # Create chunked dataset with compression
                f.create_dataset(key, 
                               data=tensor_np, 
                               chunks=chunks,
                               compression='gzip', 
                               compression_opts=9)

# Example usage:
if __name__ == "__main__":
    # Create sample tensor dictionary
    tensor_dict = {
        'tensor1': torch.randn(256, 40, 3, 64, 64),
        'tensor2': torch.randn(256, 40, 3, 64, 64),
        'tensor3': torch.randn(256, 40, 3, 64, 64)
    }

    # Save tensors
    save_tensors_to_h5(tensor_dict, 'tensors.h5')
    # Or use chunked saving for large datasets
    save_tensors_chunked(tensor_dict, 'tensors_chunked.h5')

    # Load tensors
    loaded_dict = load_tensors_from_h5('tensors.h5')