"""
Energy calculation utilities for analog crossbar operations in CrossSim.

This module provides energy estimation for analog in-memory computing operations,
taking into account device parameters, array sizes, and operation characteristics.
"""

import numpy as np
import torch
from typing import List, Dict, Any

class AnalogEnergyCalculator:
    """
    Calculate energy consumption for analog crossbar operations.
    
    Energy sources include:
    1. Static power from device leakage
    2. Dynamic power from matrix-vector multiplications  
    3. Peripheral circuit power (ADCs, DACs, buffers)
    4. Parasitic resistance power losses
    """
    
    def __init__(self, params_list: List[Any], voltage_supply: float = 1.0):
        """
        Initialize energy calculator.
        
        Args:
            params_list: List of CrossSim parameters for each layer
            voltage_supply: Supply voltage in volts
        """
        self.params_list = params_list
        self.V_supply = voltage_supply
        self.layer_energies = []
        self.total_energy = 0.0
        
        # Energy model parameters (calibrated from literature/measurements)
        self.device_params = {
            'static_power_per_device_nW': 0.1,  # Static power per device (nW)
            'resistance_on': 1e6,  # On resistance (Ohms)  
            'resistance_off': 1e9,  # Off resistance (Ohms)
            'device_capacitance_fF': 1.0,  # Device capacitance (fF)
        }
        
        self.peripheral_params = {
            'adc_energy_per_conversion_pJ': 50.0,  # ADC energy per conversion
            'dac_energy_per_conversion_pJ': 10.0,  # DAC energy per conversion
            'buffer_power_per_line_uW': 1.0,  # Buffer power per line (μW)
            'amplifier_power_per_column_uW': 5.0,  # Column amplifier power (μW)
        }
        
    def calculate_static_energy(self, array_size: tuple, time_duration_s: float) -> float:
        """
        Calculate static energy consumption from device leakage.
        
        Args:
            array_size: (rows, cols) array dimensions
            time_duration_s: Time duration in seconds
            
        Returns:
            Static energy in picojoules
        """
        rows, cols = array_size
        n_devices = rows * cols
        static_power_nW = n_devices * self.device_params['static_power_per_device_nW']
        static_energy_pJ = static_power_nW * time_duration_s * 1e9  # Convert nW*s to pJ
        return static_energy_pJ
        
    def calculate_dynamic_energy(self, array_size: tuple, input_vector: torch.Tensor, 
                                conductance_matrix: np.ndarray = None) -> float:
        """
        Calculate dynamic energy from MVM operations.
        
        Args:
            array_size: (rows, cols) array dimensions  
            input_vector: Input vector for MVM
            conductance_matrix: Optional conductance matrix for more accurate calculation
            
        Returns:
            Dynamic energy in picojoules
        """
        rows, cols = array_size
        
        if conductance_matrix is not None:
            # Use actual conductance values if available
            avg_conductance = np.mean(conductance_matrix)
        else:
            # Use average conductance estimate
            G_on = 1.0 / self.device_params['resistance_on']
            G_off = 1.0 / self.device_params['resistance_off'] 
            avg_conductance = (G_on + G_off) / 2
            
        # Calculate current flow through array
        if isinstance(input_vector, torch.Tensor):
            input_magnitude = torch.mean(torch.abs(input_vector)).item()
        else:
            input_magnitude = np.mean(np.abs(input_vector))
            
        # Energy = V^2 * G * t for each device
        # Assume operation time proportional to number of devices
        operation_time_ns = rows * 0.1  # Rough estimate: 0.1 ns per row
        
        total_current = rows * avg_conductance * input_magnitude * self.V_supply
        power_W = total_current * self.V_supply
        dynamic_energy_pJ = power_W * operation_time_ns * 1e3  # Convert W*ns to pJ
        
        return dynamic_energy_pJ
        
    def calculate_adc_energy(self, n_conversions: int, adc_bits: int) -> float:
        """
        Calculate ADC energy consumption.
        
        Args:
            n_conversions: Number of ADC conversions
            adc_bits: ADC resolution in bits
            
        Returns:
            ADC energy in picojoules
        """
        # Energy scales with resolution and number of conversions
        energy_per_conversion = self.peripheral_params['adc_energy_per_conversion_pJ']
        # Scale with resolution (exponential relationship)
        resolution_factor = 2 ** (adc_bits - 8)  # Normalized to 8-bit baseline
        total_energy = n_conversions * energy_per_conversion * resolution_factor
        return total_energy
        
    def calculate_dac_energy(self, n_conversions: int, dac_bits: int) -> float:
        """
        Calculate DAC energy consumption.
        
        Args:
            n_conversions: Number of DAC conversions
            dac_bits: DAC resolution in bits
            
        Returns:
            DAC energy in picojoules
        """
        energy_per_conversion = self.peripheral_params['dac_energy_per_conversion_pJ']
        resolution_factor = 2 ** (dac_bits - 8)  # Normalized to 8-bit baseline
        total_energy = n_conversions * energy_per_conversion * resolution_factor
        return total_energy
        
    def calculate_layer_energy(self, layer_idx: int, input_shape: tuple, 
                              weight_shape: tuple, operation_time_s: float,
                              input_data: torch.Tensor = None) -> Dict[str, float]:
        """
        Calculate total energy for a single layer.
        
        Args:
            layer_idx: Layer index
            input_shape: Input tensor shape
            weight_shape: Weight tensor shape  
            operation_time_s: Operation time in seconds
            input_data: Optional input data for more accurate calculation
            
        Returns:
            Dictionary with energy breakdown
        """
        if layer_idx >= len(self.params_list):
            return {'total': 0.0}
            
        params = self.params_list[layer_idx]
        
        # Array dimensions
        array_size = (weight_shape[0], weight_shape[1])
        
        # Static energy
        static_energy = self.calculate_static_energy(array_size, operation_time_s)
        
        # Dynamic energy
        if input_data is not None:
            dynamic_energy = self.calculate_dynamic_energy(array_size, input_data)
        else:
            # Estimate with random input
            dummy_input = torch.randn(input_shape[1:])  # Remove batch dimension
            dynamic_energy = self.calculate_dynamic_energy(array_size, dummy_input)
            
        # Peripheral energy
        adc_bits = getattr(params, 'adc_bits', 8)
        input_bits = getattr(params, 'input_bits', 8)
        
        # Number of ADC conversions (one per output)
        n_adc_conversions = weight_shape[0] * input_shape[0]  # outputs * batch_size
        adc_energy = self.calculate_adc_energy(n_adc_conversions, adc_bits)
        
        # Number of DAC conversions (one per input)  
        n_dac_conversions = weight_shape[1] * input_shape[0]  # inputs * batch_size
        dac_energy = self.calculate_dac_energy(n_dac_conversions, input_bits)
        
        # Total energy
        total_energy = static_energy + dynamic_energy + adc_energy + dac_energy
        
        energy_breakdown = {
            'static_energy_pJ': static_energy,
            'dynamic_energy_pJ': dynamic_energy, 
            'adc_energy_pJ': adc_energy,
            'dac_energy_pJ': dac_energy,
            'total_energy_pJ': total_energy,
            'total_energy_nJ': total_energy / 1000.0
        }
        
        self.layer_energies.append(energy_breakdown)
        self.total_energy += total_energy
        
        return energy_breakdown
        
    def get_total_energy_nJ(self) -> float:
        """Get total energy consumption in nanojoules."""
        return self.total_energy / 1000.0
        
    def get_energy_breakdown(self) -> Dict[str, float]:
        """Get detailed energy breakdown."""
        if not self.layer_energies:
            return {}
            
        breakdown = {
            'total_static_pJ': sum(layer['static_energy_pJ'] for layer in self.layer_energies),
            'total_dynamic_pJ': sum(layer['dynamic_energy_pJ'] for layer in self.layer_energies),
            'total_adc_pJ': sum(layer['adc_energy_pJ'] for layer in self.layer_energies),
            'total_dac_pJ': sum(layer['dac_energy_pJ'] for layer in self.layer_energies),
            'total_energy_nJ': self.get_total_energy_nJ(),
            'layer_breakdown': self.layer_energies
        }
        
        return breakdown
        
    def reset(self):
        """Reset energy counters."""
        self.layer_energies = []
        self.total_energy = 0.0
        
    def print_energy_summary(self):
        """Print a summary of energy consumption."""
        breakdown = self.get_energy_breakdown()
        
        print(f"\n=== Energy Consumption Summary ===")
        print(f"Total Energy: {breakdown['total_energy_nJ']:.3f} nJ")
        print(f"Static Energy: {breakdown['total_static_pJ']:.1f} pJ ({breakdown['total_static_pJ']/self.total_energy*100:.1f}%)")
        print(f"Dynamic Energy: {breakdown['total_dynamic_pJ']:.1f} pJ ({breakdown['total_dynamic_pJ']/self.total_energy*100:.1f}%)")
        print(f"ADC Energy: {breakdown['total_adc_pJ']:.1f} pJ ({breakdown['total_adc_pJ']/self.total_energy*100:.1f}%)")
        print(f"DAC Energy: {breakdown['total_dac_pJ']:.1f} pJ ({breakdown['total_dac_pJ']/self.total_energy*100:.1f}%)")
        print(f"===================================\n")


def estimate_inference_energy(model, params_list: List[Any], 
                            input_data: torch.Tensor, 
                            inference_time_s: float) -> Dict[str, Any]:
    """
    Estimate total energy consumption for neural network inference.
    
    Args:
        model: Neural network model
        params_list: CrossSim parameters for each layer
        input_data: Input data tensor
        inference_time_s: Total inference time in seconds
        
    Returns:
        Dictionary with energy consumption data
    """
    calculator = AnalogEnergyCalculator(params_list)
    
    # Simple approximation: distribute time evenly across layers
    time_per_layer = inference_time_s / len(params_list)
    
    # Calculate energy for each layer (simplified)
    for layer_idx, params in enumerate(params_list):
        # Estimate layer dimensions (this would need refinement for actual implementation)
        input_shape = (input_data.shape[0], 512)  # Batch size, feature dimension  
        weight_shape = (512, 512)  # Output, input dimensions
        
        calculator.calculate_layer_energy(
            layer_idx, input_shape, weight_shape, 
            time_per_layer, input_data
        )
    
    return calculator.get_energy_breakdown()
