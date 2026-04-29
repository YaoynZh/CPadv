import warnings
import matplotlib

# Set the backend for matplotlib to 'Agg' for environments without a display server
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error
from query_probability import query_one, load_ucr

# Ignore warnings to keep the output clean
warnings.filterwarnings('ignore')


def merge(intervals):
    """
    Merge overlapping or adjacent intervals.

    :param intervals: List of intervals, where each interval is a list of two integers [start, end].
    :return: Merged intervals as a list of lists.
    """
    if len(intervals) == 0:
        return []

    res = []
    intervals = list(sorted(intervals))
    low = intervals[0][0]
    high = intervals[0][1]

    for i in range(1, len(intervals)):
        if high >= intervals[i][0]:
            if high < intervals[i][1]:
                high = intervals[i][1]
        else:
            res.append([low, high])
            low = intervals[i][0]
            high = intervals[i][1]

    res.append([low, high])

    return res


def get_interval(run_tag, topk):
    """
    Retrieve and merge the top-k change point intervals.

    :param topk: The number of top change points to retrieve.
    :param run_tag: The identifier for the dataset (e.g., 'ECG200').
    :return: Merged change point intervals as a list of lists.
    """
    shaplet_pos = np.loadtxt('cp_pos/' + run_tag + '_cp_pos.txt', usecols=(2, 3))
    shaplet_pos = shaplet_pos[:topk]
    shaplet_pos = shaplet_pos.tolist()

    return merge(shaplet_pos)


def get_magnitude(run_tag, factor, normalize):
    """
    Calculate the perturbed magnitude for the time series.

    :param run_tag: The identifier for the dataset (e.g., 'ECG200').
    :param factor: The factor by which to multiply the mean magnitude to obtain the perturbed magnitude.
    :param normalize: Boolean flag to indicate whether to normalize the data.
    :return: The perturbed magnitude as a float.
    """
    data = load_ucr('data/' + run_tag + '/' + run_tag + '_attack.txt', normalize=normalize)
    X = data[:, 1:]

    max_magnitude = X.max(1)
    min_magnitude = X.min(1)
    mean_magnitude = np.mean(max_magnitude - min_magnitude)

    perturbed_mag = mean_magnitude * factor
    print('Perturbed Magnitude:', perturbed_mag)

    return perturbed_mag


class Attacker:
    """
    Class to perform adversarial attacks on time series data.
    """

    def __init__(self, run_tag, top_k, model_type, cuda, normalize, e, device):
        """
        Initialize the attacker class with necessary parameters.

        :param run_tag: The identifier for the dataset (e.g., 'ECG200').
        :param top_k: Number of top-k change points to consider.
        :param model_type: The type of model being attacked (e.g., 'f').
        :param cuda: Boolean flag indicating whether to use CUDA for computation.
        :param normalize: Boolean flag indicating whether to normalize the data.
        :param e: An additional parameter used by the model (potentially an epoch number).
        :param device: The device (CPU or GPU) to perform computations on.
        """
        self.run_tag = run_tag
        self.top_k = top_k
        self.model_type = model_type
        self.cuda = cuda
        self.intervals = get_interval(self.run_tag, self.top_k)
        self.normalize = normalize
        self.e = e
        self.device = device

    def perturb_ts(self, perturbations, ts):
        """
        Apply perturbations to a time series.

        :param perturbations: A list of perturbations corresponding to each change point interval.
        :param ts: The original time series to perturb.
        :return: The perturbed time series.
        """
        ts_tmp = np.copy(ts)
        coordinate = 0
        for interval in self.intervals:
            for i in range(int(interval[0]), int(interval[1])):
                ts_tmp[i] += perturbations[coordinate]
                coordinate += 1
        return ts_tmp

    def plot_per(self, perturbations, ts, target_class, sample_idx, prior_probs, attack_probs, factor):
        """
        Plot the original and perturbed time series for visualization.

        :param perturbations: The perturbations applied to the original time series.
        :param ts: The original time series.
        :param target_class: The target class for a targeted attack (-1 for untargeted attacks).
        :param sample_idx: The index of the sample being attacked.
        :param prior_probs: The predicted probability of the original time series.
        :param attack_probs: The predicted probability of the perturbed time series.
        :param factor: The epsilon factor used in perturbation.
        """
        # Obtain the perturbed time series
        ts_tmp = np.copy(ts)
        ts_perturbed = self.perturb_ts(perturbations=perturbations, ts=ts)

        # Start to plot
        plt.figure(figsize=(6, 4))
        plt.plot(ts_tmp, color='b', label='Original %.2f' % prior_probs)
        plt.plot(ts_perturbed, color='r', label='Perturbed %.2f' % attack_probs)
        plt.xlabel('Time', fontsize=12)

        if target_class == -1:
            plt.title('Untargeted: Sample %d, eps_factor=%.3f' %
                      (sample_idx, factor), fontsize=14)
        else:
            plt.title('Targeted(%d): Sample %d, eps_factor=%.3f' %
                      (target_class, sample_idx, factor), fontsize=14)

        plt.legend(loc='upper right', fontsize=8)
        plt.savefig('result_' + str(factor) + '_' + str(self.top_k) + '_' + str(self.model_type) + '/'
                    + self.run_tag + '/figures' + '/' + self.run_tag + '_' + str(sample_idx) + '.png')

    def fitness(self, device, perturbations, ts, sample_idx, queries, target_class=-1):
        """
        Evaluate the fitness of the perturbation for the attack optimization.

        :param device: The device (CPU or GPU) to perform computations on.
        :param perturbations: The perturbations applied to the original time series.
        :param ts: The original time series.
        :param sample_idx: The index of the sample being attacked.
        :param queries: A list containing a single integer tracking the number of queries made.
        :param target_class: The target class for a targeted attack (-1 for untargeted attacks).
        :return: The probability of the target class, which is minimized by the optimizer.
        """
        perturbations = torch.tensor(perturbations).to(device)
        queries[0] += 1

        ts_perturbed = self.perturb_ts(perturbations, ts)
        prob, _, _, _, _ = query_one(run_tag=self.run_tag, device=device, idx=sample_idx, attack_ts=ts_perturbed,
                                     target_class=target_class, normalize=self.normalize,
                                     cuda=self.cuda, model_type=self.model_type, e=self.e)
        prob = torch.tensor(prob)

        if target_class != -1:
            prob = 1 - prob

        return prob  # The fitness function is to minimize the fitness value

    def attack_success(self, device, perturbations, ts, sample_idx, iterations, target_class=-1, verbose=True):
        """
        Determine if the attack is successful and decide whether to stop early.

        :param device: The device (CPU or GPU) to perform computations on.
        :param perturbations: The perturbations applied to the original time series.
        :param ts: The original time series.
        :param sample_idx: The index of the sample being attacked.
        :param iterations: A list containing a single integer tracking the number of iterations performed.
        :param target_class: The target class for a targeted attack (-1 for untargeted attacks).
        :param verbose: Boolean flag indicating whether to print detailed logs.
        :return: Boolean indicating whether the attack is successful.
        """
        iterations[0] += 1
        print('The %d iteration' % iterations[0])
        ts_perturbed = self.perturb_ts(perturbations, ts)

        # Obtain the perturbed probability vector and the prior probability vector
        prob, prob_vector, prior_prob, prior_prob_vec, real_label = query_one(self.run_tag, device, idx=sample_idx,
                                                                              attack_ts=ts_perturbed,
                                                                              target_class=target_class,
                                                                              normalize=self.normalize,
                                                                              verbose=verbose, cuda=self.cuda,
                                                                              model_type=self.model_type,
                                                                              e=self.e)

        predict_class = torch.argmax(prob_vector).to(device)
        prior_class = torch.argmax(prior_prob_vec).to(device)
        real_label = real_label.to(device)

        # Conditions for early termination (empirical-based estimation),
        # leading to save the attacking time
        if (iterations[0] > 5 and prob > 0.99) or \
                (iterations[0] > 20 and prob > 0.9):
            print('The %d sample is not expected to successfully attack.' % sample_idx)
            print('prob: ', prob)
            return True

        if prior_class != real_label:
            print('The %d sample cannot be classified correctly, no need to attack' % sample_idx)
            return True

        if prior_class == target_class:
            print(
                'The true label of %d sample equals to target label, no need to attack' % sample_idx)
            return True

        if verbose:
            print('The Confidence of current iteration: %.4f' % prob)
            print('########################################################')

        # The criterion of attacking successfully:
        # Untargeted attack: predicted label is not equal to the original label.
        # Targeted attack: predicted label is equal to the target label.
        if ((target_class == -1 and predict_class != prior_class) or
                (target_class != -1 and predict_class == target_class)):
            print('##################### Attack Successfully! ##########################')

            return True

    def attack(self, sample_idx, device, target_class=-1, factor=0.04,
               max_iteration=50, popsize=200, verbose=True):
        """
        Perform the adversarial attack on a specific sample.

        :param sample_idx: The index of the sample to attack.
        :param device: The device (CPU or GPU) to perform computations on.
        :param target_class: The target class for a targeted attack (-1 for untargeted attacks).
        :param factor: The epsilon factor used in perturbation.
        :param max_iteration: The maximum number of iterations for the optimization algorithm.
        :param popsize: The population size for the differential evolution algorithm.
        :param verbose: Boolean flag indicating whether to print detailed logs.
        :return: The original time series, the perturbed time series, and a list of attack details.
        """
        test = load_ucr('data/' + self.run_tag + '/' + self.run_tag + '_attack.txt', normalize=self.normalize)
        ori_ts = test[sample_idx][1:]

        # Get initial predictions and probabilities
        attacked_probs, attacked_vec, prior_probs, prior_vec, real_label = query_one(self.run_tag, device,
                                                                                     idx=sample_idx,
                                                                                     attack_ts=ori_ts,
                                                                                     target_class=target_class,
                                                                                     normalize=self.normalize,
                                                                                     verbose=False)
        prior_class = torch.argmax(prior_vec).to(device)
        if prior_class != real_label:
            print('The %d sample cannot be classified correctly, no need to attack' % sample_idx)
            return ori_ts, ori_ts, [prior_probs, attacked_probs, 0, 0, 0, 0, 0, 'WrongSample']

        steps_count = 0  # Count the number of coordinates

        # Get the maximum perturbed magnitude
        perturbed_magnitude = get_magnitude(self.run_tag, factor, normalize=self.normalize)

        bounds = []
        for interval in self.intervals:
            steps_count += int(interval[1]) - int(interval[0])
            for i in range(int(interval[0]), int(interval[1])):
                bounds.append((-1 * perturbed_magnitude, perturbed_magnitude))

        print('The length of cp interval', steps_count)
        popmul = max(1, popsize // len(bounds))

        # Record the number of iterations
        iterations = [0]
        queries = [0]

        def fitness_fn(perturbations):
            return self.fitness(perturbations=perturbations, ts=ori_ts, queries=queries,
                                sample_idx=sample_idx, target_class=target_class, device=device)

        def callback_fn(x, convergence):
            return self.attack_success(perturbations=x, ts=ori_ts,
                                       sample_idx=sample_idx,
                                       iterations=iterations,
                                       target_class=target_class,
                                       verbose=verbose, device=device)

        # Perform differential evolution to find the optimal perturbations
        attack_result = differential_evolution(func=fitness_fn, bounds=bounds,
                                               maxiter=max_iteration, popsize=popmul,
                                               recombination=0.7, callback=callback_fn,
                                               atol=-1, polish=False)

        attack_ts = self.perturb_ts(attack_result.x, ori_ts)
        mse = mean_squared_error(ori_ts, attack_ts)

        # Get final predictions and probabilities after the attack
        attacked_probs, attacked_vec, prior_probs, prior_vec, real_label = query_one(self.run_tag, device,
                                                                                     idx=sample_idx,
                                                                                     attack_ts=attack_ts,
                                                                                     target_class=target_class,
                                                                                     normalize=self.normalize,
                                                                                     verbose=False)

        predicted_class = torch.argmax(attacked_vec).to(device)
        prior_class = torch.argmax(prior_vec).to(device)

        if prior_class != real_label:
            success = 'WrongSample'
        elif prior_class == target_class:
            success = 'NoNeedAttack'
        else:
            if (predicted_class.item() != prior_class.item() and target_class == -1) \
                    or (predicted_class.item() == target_class and target_class != -1):
                success = 'Success'
            else:
                success = 'Fail'

        if success == 'Success':
            self.plot_per(perturbations=attack_result.x, ts=ori_ts, target_class=target_class,
                          sample_idx=sample_idx, prior_probs=prior_probs, attack_probs=attacked_probs, factor=factor)

        return ori_ts, attack_ts, [prior_probs, attacked_probs, prior_class.item(),
                                   predicted_class.item(), queries[0], mse, iterations[0], success]