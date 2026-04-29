import numpy as np
import os
import time
import torch
from attacker import Attacker


def attack_process(opt):
    """
    Perform an adversarial attack process based on the provided options.

    Args:
        opt (dict): A dictionary containing various options for the attack process.
                    It should include:
                    - 'magnitude_factor': The factor by which to scale the perturbation.
                    - 'topk': The number of top cp pos to consider.
                    - 'model': The type of model to attack.
                    - 'run_tag': Name of the dataset.
                    - 'e': Epoch.
                    - 'cuda': Boolean indicating if CUDA should be used.
                    - 'normalize': Boolean indicating if input should be normalized.
                    - 'target_class': The class to target in the attack.
                    - 'maxitr': Maximum number of iterations for the attack.
                    - 'popsize': Population size for evolutionary strategies (if used).

    Returns:
        None
    """
    # Determine the device to use: GPU if available, otherwise CPU
    if opt['cuda']:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Create directories to store the results, including figures
    os.makedirs('result_%s_%d_%s/%s/figures' % (str(opt['magnitude_factor']), opt['k'],
                                                opt['model'], opt['run_tag']), exist_ok=True)

    # Path to the data used in the attack
    data_path = 'data/' + opt['run_tag'] + '/' + opt['run_tag'] + '_attack.txt'

    # Load the test data from the file
    test_data = np.loadtxt(data_path)
    size = test_data.shape[0] - 1  # Number of test samples
    idx_array = np.arange(size)  # Array of indices for iterating over the samples

    # Initialize the attacker with the given options
    attacker = Attacker(run_tag=opt['run_tag'], top_k=opt['k'], e=opt['e'],
                        model_type=opt['model'], cuda=opt['cuda'], normalize=opt['normalize'], device=device)

    # Record the starting time of the attack process
    start_time = time.time()

    # Initialize counters and accumulators for statistics
    success_cnt = 0
    right_cnt = 0
    total_mse = 0
    total_iterations = 0
    total_quries = 0

    # Iterate over each sample in the test data
    for idx in idx_array:
        print('###Start %s : generating adversarial example of the %d sample ###' % (opt['run_tag'], idx))

        # Perform the attack on the current sample
        ori_ts, attack_ts, info = attacker.attack(sample_idx=idx, target_class=opt['target_class'],
                                                  factor=opt['magnitude_factor'], max_iteration=opt['maxitr'],
                                                  popsize=opt['popsize'], device=device)

        # If the attack was successful, save the original and adversarial time series
        if info[-1] == 'Success':
            success_cnt += 1
            total_iterations += info[-2]
            total_mse += info[-3]
            total_quries += info[-4]

            # Save the original time series to a file
            with open('result_' + str(opt['magnitude_factor']) + '_' + str(opt['k']) + '_' + opt['model']
                      + '/' + opt['run_tag'] + '/ori_time_series.txt', 'a+') as file0:
                file0.write('%d %d ' % (idx, info[3]))
                for i in ori_ts:
                    file0.write('%.4f ' % i)
                file0.write('\n')

            # Save the adversarial time series to a file
            with open('result_' + str(opt['magnitude_factor']) + '_' + str(opt['k']) + '_' + opt['model']
                      + '/' + opt['run_tag'] + '/attack_time_series.txt', 'a+') as file:
                file.write('%d %d ' % (idx, info[3]))
                for i in attack_ts:
                    file.write('%.4f ' % i)
                file.write('\n')

        # Count the correctly classified samples
        if info[-1] != 'WrongSample':
            right_cnt += 1

        # Save the attack information (whether successful or not)
        with open('result_' + str(opt['magnitude_factor']) + '_' + str(opt['k']) + '_' + opt['model']
                  + '/' + opt['run_tag'] + '/information.txt', 'a+') as file:
            file.write('%d ' % idx)
            for i in info:
                if isinstance(i, int):
                    file.write('%d ' % i)
                elif isinstance(i, float):
                    file.write('%.4f ' % i)
                else:
                    file.write(str(i) + ' ')
            file.write('\n')

    # Record the end time of the attack process
    endtime = time.time()
    total = endtime - start_time  # Total running time of the process

    # Print useful information and statistics
    print('Running time: %.4f ' % total)
    print('Correctly-classified samples: %d' % right_cnt)
    print('Successful samples: %d' % success_cnt)
    print('Success rate：%.2f%%' % (success_cnt / right_cnt * 100))
    print('Misclassification rate：%.2f%%' % (success_cnt / size * 100))
    print('ANI: %.2f' % (total_iterations / success_cnt))
    print('MSE: %.4f' % (total_mse / success_cnt))
    print('Mean queries：%.2f\n' % (total_quries / success_cnt))

    # Save the statistics and running time to a file
    with open('result_' + str(opt['magnitude_factor']) + '_' + str(opt['k']) + '_' + opt['model']
              + '/' + opt['run_tag'] + '/information.txt', 'a+') as file:
        file.write('Running time:%.4f\n' % total)
        file.write('Correctly-classified samples: %d\n' % right_cnt)
        file.write('Successful samples:%d\n' % success_cnt)
        file.write('Success rate：%.2f%%\n' % (success_cnt / right_cnt * 100))
        file.write('Misclassification rate：%.2f%%\n' % (success_cnt / size * 100))
        file.write('ANI:%.2f\n' % (total_iterations / success_cnt))
        file.write('MSE:%.4f\n' % (total_mse / success_cnt))
        file.write('Mean queries：%.2f\n' % (total_quries / success_cnt))

