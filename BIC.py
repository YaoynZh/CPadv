import os
from sklearn.mixture import GaussianMixture
import numpy as np
from query_probability import load_ucr
import ruptures as rpt
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def detect_change_points(data, length, model="l2", pen=5):
    """
    Detect change points in the given time series data using the PELT algorithm.

    Parameters:
        data (array-like): The time series data to analyze.
        length: the length of time series data
        model (str): The cost model to use for change point detection (default is "l2").
        pen (int): The penalty value to control the number of detected change points (default is 5).

    Returns:
        list: Indices of the detected change points (excluding the last one).
    """
    if length > 500:
        algo = rpt.Pelt(model=model, min_size=1, jump=1).fit(data)
    else:
        algo = rpt.Binseg(model=model, min_size=1, jump=1).fit(data) #It is used in situations where it is desired to control the number of change points
    bkps = algo.predict(pen=pen)
    return bkps[:-1]  # Return all change points except the last one


def merge_intervals(intervals):
    """
    Merge overlapping intervals.

    Parameters:
        intervals (list of tuples): A list of intervals where each interval is represented as a tuple (start, end).

    Returns:
        list: A list of non-overlapping intervals.
    """
    if not intervals:
        return []

    # Sort intervals by the start value
    intervals.sort(key=lambda x: x[0])

    merged = [intervals[0]]
    for current_start, current_end in intervals[1:]:
        last_end = merged[-1][1]

        # Check if the current interval overlaps with the last merged interval
        if current_start <= last_end:
            merged[-1] = (merged[-1][0], max(last_end, current_end))
        else:
            merged.append((current_start, current_end))

    return merged


def evaluate_fit_elbow_method(thresholds, data, pdf):
    """
    Evaluate the fit using the Elbow Method to select the best threshold.

    Parameters:
    thresholds (list of float): List of candidate thresholds.
    data (np.array): The original data points.
    pdf (np.array): The probability density function values of the data.

    Returns:
    float: The threshold corresponding to the elbow point.
    """
    errors = []

    for threshold_factor in thresholds:
        high_density_intervals = []

        for peak in peaks:
            peak_density = pdf[peak]
            threshold = peak_density * threshold_factor
            left = right = peak

            # Extend the interval to the left
            while left > 0 and pdf[left] > threshold:
                left -= 1

            # Extend the interval to the right
            while right < len(pdf) - 1 and pdf[right] > threshold:
                right += 1

            # Store the interval
            interval = (x_range[left][0], x_range[right][0])
            high_density_intervals.append(interval)

        # Calculate the total error for this threshold
        total_error = 0
        for interval in high_density_intervals:
            start, end = interval
            segment = data[(data >= start) & (data <= end)]
            if len(segment) > 0:
                mean = np.mean(segment)
                total_error += np.sum((segment - mean) ** 2)

        errors.append(total_error)

    # Identify the elbow point
    elbow_point = identify_elbow_point(thresholds, errors)

    return elbow_point


def identify_elbow_point(thresholds, errors):
    """
    Identify the elbow point in the errors vs. thresholds curve.

    Parameters:
    thresholds (list of float): List of candidate thresholds.
    errors (list of float): Corresponding errors for each threshold.

    Returns:
    float: The threshold at the elbow point.
    """
    # Normalize thresholds and errors
    norm_thresholds = (thresholds - np.min(thresholds)) / (np.max(thresholds) - np.min(thresholds))
    norm_errors = (errors - np.min(errors)) / (np.max(errors) - np.min(errors))

    # Assume the first and last points define a line
    p1 = np.array([norm_thresholds[0], norm_errors[0]])
    p2 = np.array([norm_thresholds[-1], norm_errors[-1]])

    # Calculate the distances of all points from the line
    distances = []
    for i in range(len(norm_thresholds)):
        p = np.array([norm_thresholds[i], norm_errors[i]])
        distance = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
        distances.append(distance)

    # The point with the maximum distance is the elbow point
    max_distance_idx = np.argmax(distances)
    return thresholds[max_distance_idx]

if __name__ == '__main__':
    # Define the dataset tag and file path
    run_tag = 'ECG200'
    path = 'data/' + run_tag + '/' + run_tag + '_cp.txt'

    # Load the dataset
    data = load_ucr(path)
    data = data[:, 1:]  # Remove the first column (assuming it's an index or label)
    _, length = data.shape

    # Detect change points for each time series in the dataset
    all_bkps = [detect_change_points(ts,length) for ts in data]

    # Flatten the list of breakpoints into a single array
    all_bkps_flat = np.concatenate(all_bkps).reshape(-1, 1)

    # Plot the distribution of detected change points
    y = np.zeros_like(all_bkps_flat)
    plt.scatter(all_bkps_flat, y, alpha=0.5, color='red')
    plt.title("Distribution of Change Points")
    plt.xlabel("Time")
    plt.show()

    # Determine the optimal number of Gaussian components using BIC
    n_components = np.arange(1, 15)
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(all_bkps_flat) for n in n_components]
    bics = [m.bic(all_bkps_flat) for m in models]
    min_index = np.argmin(bics) + 1  # Get the index of the model with the lowest BIC
    plt.plot(n_components, bics, label='BIC')
    plt.legend()
    plt.xlabel('Number of components')
    plt.ylabel('Information Criterion')
    plt.show()

    # Fit the Gaussian Mixture Model with the optimal number of components
    gmm = GaussianMixture(n_components=min_index)
    gmm.fit(all_bkps_flat.reshape(-1, 1))

    # Generate a range of values for evaluating the GMM
    x_range = np.linspace(min(all_bkps_flat), max(all_bkps_flat), 1000).reshape(-1, 1)
    logprob = gmm.score_samples(x_range)
    pdf = np.exp(logprob)  # Calculate the probability density function



    # Identify the peaks in the density function
    peaks, _ = find_peaks(pdf, height=0)
    high_density_intervals = []

    best_threshold = None
    best_fit_score = float('inf')  # or -inf if you're maximizing

    thresholds = [0.5 + 0.02 * i for i in range(26)]  # Thresholds from 0.5 to 1.0
    best_threshold = evaluate_fit_elbow_method(thresholds, data, pdf)

    # Use the best_threshold for the final peak selection and interval extraction

    print('best T',best_threshold)


    for peak in peaks:
        peak_density = pdf[peak]
        threshold = peak_density * best_threshold  # Define the threshold for peak selection
        left = right = peak

        # Extend the interval to the left
        while left > 0 and pdf[left] > threshold:
            left -= 1

        # Extend the interval to the right
        while right < len(pdf) - 1 and pdf[right] > threshold:
            right += 1

        # Store the interval
        interval = (x_range[left][0], x_range[right][0])
        high_density_intervals.append(interval)

    # Merge overlapping intervals
    non_overlapping_intervals = merge_intervals(high_density_intervals)
    print(high_density_intervals)

    # Save the intervals to a file
    file_path = 'cp_pos/' + run_tag + '_cp_pos.txt'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as file:
        for i, (pos1, pos2) in enumerate(high_density_intervals):
            int1 = int(pos1)
            int2 = int(pos2)
            file.write(f"{run_tag} {i} {int1} {int2}\n")

    # Plot the density estimation and the identified high-density intervals
    plt.title("Gaussian Mixture Model Density Estimation")
    plt.xlabel("Time")
    plt.plot(x_range.flatten(), pdf, label='GMM Density')
    for start, end in non_overlapping_intervals:
        plt.axvspan(start, end, color='blue', alpha=0.3)
    plt.legend()
    plt.show()
