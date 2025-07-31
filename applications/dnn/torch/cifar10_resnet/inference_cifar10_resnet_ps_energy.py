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
from simulator import CrossSimParameters
from simulator.algorithms.dnn.torch.convert import from_torch, convertible_modules
from find_adc_range import find_adc_range
from dnn_inference_params import dnn_inference_params
from energy_calculator import AnalogEnergyCalculator

# Energy calculation helper class
class EnergyTracker:
    def __init__(self, params_list, n_layers):
        self.params_list = params_list
        self.n_layers = n_layers
        self.total_energy = 0.0
        self.layer_energies = np.zeros(n_layers)
        
        # Use the more sophisticated energy calculator
        self.energy_calculator = AnalogEnergyCalculator(params_list)
        
    def estimate_batch_energy(self, inputs, inference_time_s):
        """Estimate energy for a batch of inputs"""
        batch_size = inputs.shape[0]
        
        # Reset calculator for this batch
        self.energy_calculator.reset()
        
        # Simplified layer-wise energy calculation
        # In practice, you'd want to hook into the actual layer operations
        for layer_idx in range(self.n_layers):
            # Estimate layer dimensions based on ResNet architecture
            if layer_idx == 0:  # First conv layer
                input_shape = (batch_size, 3, 32, 32)
                weight_shape = (16, 3, 3, 3)  # Simplified
            elif layer_idx < self.n_layers - 1:  # Conv layers
                input_shape = (batch_size, 16, 32, 32)  
                weight_shape = (16, 16, 3, 3)
            else:  # Final FC layer
                input_shape = (batch_size, 512)
                weight_shape = (10, 512)
                
            # Calculate energy for this layer
            layer_energy = self.energy_calculator.calculate_layer_energy(
                layer_idx, input_shape, weight_shape, 
                inference_time_s / self.n_layers, inputs
            )
            
        return self.energy_calculator.get_total_energy_nJ()
        
    def get_total_energy_nJ(self):
        """Return total energy in nanojoules"""
        return self.energy_calculator.get_total_energy_nJ()
    
    def reset(self):
        """Reset energy counters"""
        self.energy_calculator.reset()
        self.total_energy = 0.0
        self.layer_energies.fill(0.0)

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
    'adc_bits': 0, # set to 0 to disable adc_range
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
alpha_pairs = [(1.0, 1.0)] #(1.0, 1.0), (1.0, 0.0), (0.0, 1.0)
Nslices = [1,2,4,8] #1, 2, 4, 8 (default 1)
input_slice_size = [1] #1, 2, 4, 8 (default cal 1)
TID_amounts = [0,10,20,50,200,500,1500] # 0, 10, 20, 50, 200, 500, 1500
adc_range_option = ["CALIBRATED"] #"CALIBRATED", "MAX", "GRANULAR"
sweep_values = [core_styles, Nslices, TID_amounts, adc_range_option, alpha_pairs, input_slice_size]

results = []
for combo in itertools.product(*sweep_values):
    core_style, Nslice, TID_amount, adc_range_option, (alpha_mu, alpha_sig), input_slice_size = combo
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
    
    # Initialize energy tracker
    energy_tracker = EnergyTracker(params_list, n_layers)

    # Setup data loader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = datasets.CIFAR10(root='./', train=False, download=True,
                               transform=transforms.Compose([transforms.ToTensor(), normalize]))
    dataset = torch.utils.data.Subset(dataset, np.arange(N))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    accuracies = np.zeros(Nruns)
    energy_consumptions = np.zeros(Nruns)
    for m in range(Nruns):
        energy_tracker.reset()
        T1 = time.time()
        y_pred = np.zeros(N); y = np.zeros(N); k_idx = 0
        
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            
            # Time the inference for energy estimation
            inference_start = time.time()
            output = analog_resnet(inputs)
            inference_end = time.time()
            inference_time_s = inference_end - inference_start
            
            output = output.to(device)
            y_pred_k = output.data.cpu().numpy()
            
            # Estimate energy consumption for this batch
            batch_energy_nJ = energy_tracker.estimate_batch_energy(inputs, inference_time_s)
            
            # accumulate predictions
            bs = y_pred_k.shape[0]
            y_pred[k_idx:k_idx+bs] = y_pred_k.argmax(axis=1)
            y[k_idx:k_idx+bs] = labels.cpu().numpy()
            k_idx += bs
            if print_progress:
                acc = 100 * np.sum(y[:k_idx] == y_pred[:k_idx]) / k_idx
                energy_nJ = energy_tracker.get_total_energy_nJ()
                print(f"Image {k_idx}/{N}, accuracy so far = {acc:.2f}%, energy = {energy_nJ:.3f} nJ", end='\r')
        T2 = time.time()
        top1 = np.mean(y == y_pred)
        total_energy_nJ = energy_tracker.get_total_energy_nJ()
        accuracies[m] = top1
        energy_consumptions[m] = total_energy_nJ
        print(f"\nInference finished. Elapsed time: {T2-T1:.3f} sec")
        print(f"Accuracy: {top1*100:.2f}% ({int(top1*N)}/{N})")
        print(f"Total Energy Consumption: {total_energy_nJ:.3f} nJ")
        print(f"Energy per Image: {total_energy_nJ/N:.3f} nJ/image")
        print(f"Energy Efficiency: {top1*N/total_energy_nJ:.2f} correct classifications/nJ")
        
        # Print detailed energy breakdown
        energy_tracker.energy_calculator.print_energy_summary()
        if m < Nruns - 1:
            from simulator.algorithms.dnn.torch.convert import reinitialize
            reinitialize(analog_resnet)
    if Nruns > 1:
        print("==========")
        print(f"Mean accuracy:  {100*np.mean(accuracies):.2f}%")
        print(f"Stdev accuracy: {100*np.std(accuracies):.2f}%")
        print(f"Mean energy consumption: {np.mean(energy_consumptions):.3f} nJ")
        print(f"Stdev energy consumption: {np.std(energy_consumptions):.3f} nJ")

    # Record final accuracy and energy
    rec = {
        'core_style': core_style,
        'Nslices': Nslice,
        'input_slice_size': input_slice_size,
        'TID_amount': TID_amount,
        'alpha_mu': alpha_mu,
        'alpha_sig': alpha_sig,
        'accuracy': float(top1),
        'energy_consumption_nJ': float(total_energy_nJ),
        'energy_per_image_nJ': float(total_energy_nJ/N),
        'energy_efficiency_correct_per_nJ': float(top1*N/total_energy_nJ) if total_energy_nJ > 0 else 0.0,
        'inference_time_s': float(T2-T1)
    }

    df = pd.DataFrame([rec])

    df.to_csv(
        "results_wbit_noADC_7_25_25_1440_nrg.csv",
        mode='a',
        header=not os.path.exists("results_wbit_noADC_7_25_25_1440_nrg.csv"),
        index=False
    )

    print(f"combo={combo} → accuracy={top1:.4f}, energy={total_energy_nJ:.3f} nJ, results flushed to disk")

    # free resources
    del analog_resnet, params_list
    torch.cuda.empty_cache()
    if cp:
        cp.get_default_memory_pool().free_all_blocks()
