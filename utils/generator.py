# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 20:51:22 2025

@author: User
"""
from data.group_dataset import DMU_group
from data.compined_dataset import Group_collection
from efficiency_scores import calculate_efficiencies



def dataset_creation(n_groups, n_dmus, n_inputs, n_outputs, input_range, output_range):
    dataset = Group_collection()
    for group_index in range(n_groups):
        current_dataset = DMU_group(n_dmus, n_inputs, n_outputs, input_range, output_range)
        current_dataset.data_generator()
        
        #calculate efficiencies
        for dmu_index in range(n_dmus):
            theta, s_minus, s_plus, lambdas = calculate_efficiencies(current_dataset.inputs, current_dataset.outputs, dmu_index)
            current_dataset.efficiencies.append(round(theta.item(),6)) 
        
        dataset.add_group(current_dataset)
        
    return dataset