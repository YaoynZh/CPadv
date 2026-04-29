from attack import attack_process

if __name__ == "__main__":
    config = {
        'cuda': True,
        'target_class': -1,
        'popsize': 1,
        'magnitude_factor': 0.04,
        'maxitr': 50,
        'run_tag': 'ECG200',
        'model': 'f',
        'k': 6,
        'normalize': True,
        'e': 1499
    }
    attack_process(config)
