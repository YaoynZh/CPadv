import numpy as np
import torch
import torch.nn as nn

def load_ucr(path, normalize=False):
    """
    Loads and optionally normalizes the UCR dataset from the specified path.

    Args:
        path (str): The path to the dataset file.
        normalize (bool): Whether to normalize the dataset or not. Default is False.

    Returns:
        np.ndarray: The loaded dataset with the first column as labels and the rest as features.
    """
    data = np.loadtxt(path)
    data[:, 0] -= 1  # Adjust label indices to start from 0

    # Ensure that labels are within [0, num_classes-1] range
    num_classes = len(np.unique(data[:, 0]))
    for i in range(data.shape[0]):
        if data[i, 0] < 0:
            data[i, 0] = num_classes - 1

    # Normalize the dataset if requested
    if normalize:
        mean = data[:, 1:].mean(axis=1, keepdims=True)
        std = data[:, 1:].std(axis=1, keepdims=True)
        data[:, 1:] = (data[:, 1:] - mean) / (std + 1e-8)

    return data

def query_one(run_tag, device, idx, attack_ts, target_class=-1, normalize=False,
              e=1499, verbose=False, cuda=True, model_type='r'):
    """
    Queries the pre-trained model with a given time series to evaluate the effectiveness of an attack.

    Args:
        run_tag (str): Identifier for the run, used for paths.
        device (torch.device): Device to perform computation on (CPU or CUDA).
        idx (int): Index of the test sample to be used.
        attack_ts (np.ndarray): Time series data representing the attack.
        target_class (int): Target class for the attack. Default is -1 (use true label).
        normalize (bool): Whether to normalize the dataset or not. Default is False.
        e (int): Epoch or other identifier (not used directly in this function).
        verbose (bool): Whether to print detailed information. Default is False.
        cuda (bool): Whether to use CUDA. Default is True.
        model_type (str): Type of model, default is 'r' (not directly used in the function).

    Returns:
        tuple: (prob2, prob_vector2, prob, prob_vector, real_label)
            - prob2: Probability of the target class after the attack.
            - prob_vector2: Probability distribution after the attack.
            - prob: Initial probability of the true label before the attack.
            - prob_vector: Initial probability distribution before the attack.
            - real_label: The actual label of the sample.
    """
    # Convert attack time series to a torch tensor and move to the specified device
    ts = torch.from_numpy(attack_ts).float().to(device)

    # Load test data from the specified path
    data_path = 'data/' + run_tag + '/' + run_tag + '_attack.txt'
    test_data = load_ucr(path=data_path, normalize=normalize)
    test_data = torch.from_numpy(test_data).to(device)
    Y = test_data[:, 0]  # Extract labels

    # Determine the number of classes from the unique labels
    n_class = torch.unique(Y).size(0)
    test_one = test_data[idx]  # Select the sample by index

    # Extract features and label from the selected sample
    X = test_one[1:].float()
    y = test_one[0].long()
    y = y.to(device)

    real_label = y  # Store the real label for reference

    # If a target class is specified, use it instead of the real label
    if target_class != -1:
        y = target_class

    # Move tensors to the specified device
    ts = ts.to(device)
    X = X.to(device)

    # Load the pre-trained model
    model_path = 'model_checkpoints/' + run_tag + '/pre_fTrained.pth'
    model = torch.load(model_path, map_location=device)

    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        softmax = nn.Softmax(dim=-1)

        # Get the probability distribution for the original sample
        out = model(X)
        prob_vector = softmax(out)
        prob = prob_vector.view(n_class)[y].item()

        # Get the probability distribution for the attacked sample
        out2 = model(ts)
        prob_vector2 = softmax(out2)
        prob2 = prob_vector2.view(n_class)[y].item()

        if verbose:
            print('Target_Class：', target_class)
            print(f'Prior Confidence of the {idx} sample is  {prob:.4f} ')

    return prob2, prob_vector2, prob, prob_vector, real_label
