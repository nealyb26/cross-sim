"""
Parameterizable inference simulation script for CIFAR-10 ResNets 
with parameter sweep capability.
"""

import torch
import os
import itertools
import pandas as pd
from torchvision import datasets, transforms
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
import warnings, sys, time
from build_resnet_cifar10 import ResNet_cifar10
warnings.filterwarnings('ignore')
sys.path.append("../../")  # to import dnn_inference_params
sys.path.append("../../../../")  # to import simulator
from simulator.algorithms.dnn.torch.convert import from_torch, convertible_modules
from find_adc_range import find_adc_range
from dnn_inference_params import dnn_inference_params

# Depth parameter for model selection
n = 3  # ResNet-20
useGPU = True
N = 100
batch_size = 32
Nruns = 1
print_progress = True

depth = 6 * n + 2
print(f"Model: ResNet-{depth}")
print("CIFAR-10: using " + ("GPU" if useGPU else "CPU"))
print(f"Number of images: {N}")
print(f"Number of runs: {Nruns}")
print(f"Batch size: {batch_size}")
device = torch.device("cuda:0" if (torch.cuda.is_available() and useGPU) else "cpu")

# Temp Code to be Removed Start
existing_results = pd.read_csv("results_ibit_wbit_c_7_29_25_2220.csv")

def combo_exists(core_style, Nslice, input_slice_size, TID_amount, alpha_mu, alpha_sig):
    return not existing_results[
        (existing_results['core_style'] == core_style) &
        (existing_results['Nslices'] == Nslice) &
        (existing_results['input_slice_size'] == input_slice_size) &
        (existing_results['TID_amount'] == TID_amount) &
        (existing_results['alpha_mu'] == alpha_mu) &
        (existing_results['alpha_sig'] == alpha_sig)
    ].empty
# Temp Code End

# Load PyTorch model once
resnet_model = ResNet_cifar10(n).to(device)
resnet_model.load_state_dict(
    torch.load(f'./models/resnet{depth}_cifar10.pth', map_location=device)
)
resnet_model.eval()
n_layers = len(convertible_modules(resnet_model))
print(f"# of layers {n_layers}")

# Common base args
base_params_args = {
    'ideal': False,
    'core_style': "BALANCED",
    'Nslices': 1,
    'weight_bits': 8,
    'weight_percentile': 100,
    'digital_bias': True,
    'Rmin': 62500,
    'Rmax': 62500000,
    'infinite_on_off_ratio': False,
    'error_model': "customSONOS",
    'alpha_error': 0.0,
    'TID_amount': 0,
    'shift_csv_loc': "/home/bagain/aimc_testbed/examples/sonos_current_shift.csv",
    'std_csv_loc': "/home/bagain/aimc_testbed/examples/sonos_current_std.csv",
    'alpha_mu': 1.0,
    'alpha_sig': 1.0,
    'proportional_error': False,
    'noise_model': "SONOS",
    'alpha_noise': 0.0,
    'proportional_noise': False,
    'drift_model': "SONOS",
    't_drift': 0,
    'NrowsMax': 1152,
    'NcolsMax': None,
    'Rp_row': 0,
    'Rp_col': 0,
    'interleaved_posneg': False,
    'subtract_current_in_xbar': True,
    'current_from_input': True,
    'input_bits': 8,
    'input_bitslicing': False,
    'input_slice_size': 8,
    'adc_bits': 8, # set to 0 to disable adc_range
    'adc_range_option': "CALIBRATED",
    'adc_type': "generic",
    'adc_per_ibit': False,
    'useGPU': useGPU
}

# Precomputed limits
input_ranges = np.load(f"./calibrated_config/input_limits_ResNet{depth}.npy")
adc_ranges = find_adc_range(base_params_args, n_layers, depth)

# Sweep values
# (default) means that calibration available in original CS-3.1.1 repository
core_styles = ["BALANCED"] #"BALANCED", "OFFSET" (default cal "BALANCED")
alpha_pairs = [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0)] #(1.0, 1.0), (1.0, 0.0), (0.0, 1.0)
Nslices = [4,8] #1, 2, 4, 8 (default 1)
input_slice_size = [1,2,4,8] #1, 2, 4, 8 (default cal 1)
TID_amounts = [0,10,15,20,50,100,200,500,1000,1500] # 0, 10, 20, 50, 200, 500, 1500
adc_range_option = ["CALIBRATED"] #"CALIBRATED", "MAX", "GRANULAR"
sweep_values = [core_styles, Nslices, TID_amounts, adc_range_option, alpha_pairs, input_slice_size]

results = []
for combo in itertools.product(*sweep_values):
    core_style, Nslice, TID_amount, adc_range_option, (alpha_mu, alpha_sig), input_slice_size = combo
    
    # Temp Code to be removed. Start
    if combo_exists(core_style, Nslice, input_slice_size, TID_amount, alpha_mu, alpha_sig):
        print(f"Skipping already completed combo: {combo}")
        continue
    # Temp Code End
    
    input_bitslicing = (input_slice_size < base_params_args['input_bits'])
    # Build this iteration's params
    this_params = base_params_args.copy()
    this_params.update({
        'core_style': core_style,
        'Nslices': Nslice,
        'TID_amount': TID_amount,
        'alpha_mu': alpha_mu,
        'alpha_sig': alpha_sig,
        'adc_range_option' : adc_range_option, 
        'input_slice_size': input_slice_size,
        'input_bitslicing': input_bitslicing
    })
    subset_keys = ['TID_amount', 'input_slice_size', 'Nslices', 'alpha_mu', 'alpha_sig', 'core_style', 'adc_range_option']
    subset = {k: this_params[k] for k in subset_keys if k in this_params}
    print("Running:", subset)
    
    adc_ranges = find_adc_range(this_params, n_layers, depth)
    # print("ADC range loaded with following shape:")
    # print(adc_ranges.shape)

    # Build layer-wise params_list
    params_list = [None] * n_layers
    for k in range(n_layers):
        p_args = this_params.copy()
        p_args['positiveInputsOnly'] = (False if k == 0 else True)
        p_args['input_range'] = input_ranges[k]
        p_args['adc_range'] = adc_ranges[k]
        params_list[k] = dnn_inference_params(**p_args)
    # Convert PyTorch model to analog for this sweep combo
    analog_resnet = from_torch(resnet_model, params_list, fuse_batchnorm=True, bias_rows=0)

    # Setup data loader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = datasets.CIFAR10(root='./', train=False, download=True,
                               transform=transforms.Compose([transforms.ToTensor(), normalize]))
    dataset = torch.utils.data.Subset(dataset, np.arange(N))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    accuracies = np.zeros(Nruns)
    for m in range(Nruns):
        T1 = time.time()
        y_pred = np.zeros(N); y = np.zeros(N); k_idx = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            output = analog_resnet(inputs)
            output = output.to(device)
            y_pred_k = output.data.cpu().numpy()
            # accumulate predictions
            bs = y_pred_k.shape[0]
            y_pred[k_idx:k_idx+bs] = y_pred_k.argmax(axis=1)
            y[k_idx:k_idx+bs] = labels.cpu().numpy()
            k_idx += bs
            if print_progress:
                acc = 100 * np.sum(y[:k_idx] == y_pred[:k_idx]) / k_idx
                print(f"Image {k_idx}/{N}, accuracy so far = {acc:.2f}%", end='\r')
        T2 = time.time()
        top1 = np.mean(y == y_pred)
        accuracies[m] = top1
        print(f"\nInference finished. Elapsed time: {T2-T1:.3f} sec")
        print(f"Accuracy: {top1*100:.2f}% ({int(top1*N)}/{N})\n")
        if m < Nruns - 1:
            from simulator.algorithms.dnn.torch.convert import reinitialize
            reinitialize(analog_resnet)
    if Nruns > 1:
        print("==========")
        print(f"Mean accuracy:  {100*np.mean(accuracies):.2f}%")
        print(f"Stdev accuracy: {100*np.std(accuracies):.2f}%")

    # Record final accuracy
    rec = {
        'core_style': core_style,
        'Nslices': Nslice,
        'input_slice_size': input_slice_size,
        'TID_amount': TID_amount,
        'alpha_mu': alpha_mu,
        'alpha_sig': alpha_sig,
        'accuracy': float(top1)
    }

    df = pd.DataFrame([rec])

    df.to_csv(
        "results_ibit_wbit_c_7_30_25_1200.csv",
        mode='a',
        header=not os.path.exists("results_ibit_wbit_c_7_30_25_1200.csv"),
        index=False
    )

    print(f"combo={combo} → accuracy={top1:.4f}, results flushed to disk")

    # free resources
    del analog_resnet, params_list
    torch.cuda.empty_cache()
    if cp:
        cp.get_default_memory_pool().free_all_blocks()
