import numpy as np
import pandas as pd
import json
from pathlib import Path
from sonata.circuit import File
from sonata.reports.spike_trains import SpikeTrains
import pygenn
import matplotlib.pyplot as plt
import pickle

GLIF3 = pygenn.genn_model.create_custom_neuron_class(
    "GLIF3",
    param_names=[
        "C",
        "G",
        "El",
        "spike_cut_length",
        "th_inf",
        "V_reset",
        "asc_amp_array_1",
        "asc_amp_array_2",
        "asc_stable_coeff_1",
        "asc_stable_coeff_2",
        "asc_decay_rates_1",
        "asc_decay_rates_2",
        "asc_refractory_decay_rates_1",
        "asc_refractory_decay_rates_2",
    ],
    var_name_types=[
        ("V", "double"),
        ("refractory_countdown", "int"),
        ("ASC_1", "scalar"),
        ("ASC_2", "scalar"),
    ],
    sim_code="""

    // Sum after spike currents
    double sum_of_ASC = $(ASC_1)*$(asc_stable_coeff_1) + $(ASC_2)*$(asc_stable_coeff_2);

    // Voltage
    if ($(refractory_countdown) <= 0) {
        $(V)+=1/$(C)*($(Isyn)+sum_of_ASC-$(G)*($(V)-$(El)))*DT;
    }

    // ASCurrents
    if ($(refractory_countdown) <= 0) {
        $(ASC_1) *= $(asc_decay_rates_1);
        $(ASC_2) *= $(asc_decay_rates_2);
        }


    // Decrement refractory_countdown by 1; Do not decrement past -1
    if ($(refractory_countdown) > -1) {
        $(refractory_countdown) -= 1;
    }
    """,
    threshold_condition_code="$(V) > $(th_inf)",
    reset_code="""
    $(V)=$(V_reset);
    $(ASC_1) = $(asc_amp_array_1) + $(ASC_1) * $(asc_refractory_decay_rates_1);
    $(ASC_2) = $(asc_amp_array_2) + $(ASC_2) * $(asc_refractory_decay_rates_2);
    $(refractory_countdown) = $(spike_cut_length);
    """,
)

psc_Alpha = pygenn.genn_model.create_custom_postsynaptic_class(
    class_name="Alpha",
    decay_code="""
    $(x) = exp(-DT/$(tau)) * ((DT * $(inSyn) * exp(1.0f) / $(tau)) + $(x));
    $(inSyn)*=exp(-DT/$(tau));
    """,
    apply_input_code="""
    $(Isyn) += $(x);     
""",
    var_name_types=[("x", "scalar")],
    param_names=[("tau")],
)


def spikes_list_to_start_end_times(spikes_list):

    spike_counts = [len(n) for n in spikes_list]

    # Get start and end indices of each spike sources section
    end_spike = np.cumsum(spike_counts)
    start_spike = np.empty_like(end_spike)
    start_spike[0] = 0
    start_spike[1:] = end_spike[0:-1]

    spike_times = np.hstack(spikes_list)
    return start_spike, end_spike, spike_times


def get_dynamics_params(
    node_types_df, dynamics_base_dir, sim_config, node_dict, model_name
):
    dynamics_file = node_types_df[node_types_df["model_name"] == model_name][
        "dynamics_params"
    ].iloc[0]
    dynamics_path = Path(dynamics_base_dir, dynamics_file)
    with open(dynamics_path) as f:
        old_dynamics_params = json.load(f)
    num_neurons = len(node_dict[model_name])

    DT = sim_config["run"]["dt"]
    asc_decay = np.array(old_dynamics_params["asc_decay"])
    r = np.array([1.0, 1.0])  # NEST default
    t_ref = old_dynamics_params["t_ref"]
    asc_decay_rates = np.exp(-asc_decay * sim_config["run"]["dt"])
    asc_stable_coeff = (1.0 / asc_decay / DT) * (1.0 - asc_decay_rates)
    asc_refractory_decay_rates = r * np.exp(-asc_decay * t_ref)
    
    assert len(old_dynamics_params["asc_init"]) == 2
    assert len(old_dynamics_params["asc_amps"]) == 2
    assert len(asc_refractory_decay_rates) == 2
    
    dynamics_params_renamed = {
        "C": old_dynamics_params["C_m"] / 1000,  # pF -> nF
        "G": old_dynamics_params["g"] / 1000,  # nS -> uS
        "El": old_dynamics_params["E_L"],
        "th_inf": old_dynamics_params["V_th"],
        "dT": DT,
        "V": old_dynamics_params["V_m"],
        "spike_cut_length": round(old_dynamics_params["t_ref"] / DT),
        "refractory_countdown": -1,
        "V_reset": old_dynamics_params["V_reset"],  # BMTK rounds to 3rd decimal
        "ASC_1": old_dynamics_params["asc_init"][0] / 1000,  # pA -> nA
        "ASC_2": old_dynamics_params["asc_init"][1] / 1000,  # pA -> nA
        "asc_stable_coeff": asc_stable_coeff,
        "asc_decay_rates": asc_decay_rates,
        "asc_refractory_decay_rates": asc_refractory_decay_rates,
        "asc_amp_array_1": old_dynamics_params["asc_amps"][0] / 1000,  # pA->nA
        "asc_amp_array_2": old_dynamics_params["asc_amps"][1] / 1000,  # pA->nA
        "asc_stable_coeff_1": asc_stable_coeff[0],
        "asc_stable_coeff_2": asc_stable_coeff[1],
        "asc_decay_rates_1": asc_decay_rates[0],
        "asc_decay_rates_2": asc_decay_rates[1],
        "asc_refractory_decay_rates_1": asc_refractory_decay_rates[0],
        "asc_refractory_decay_rates_2": asc_refractory_decay_rates[1],
        "tau_syn": old_dynamics_params["tau_syn"],
    }

    return dynamics_params_renamed, num_neurons


def construct_populations(
    model,
    pop_dict,
    all_model_names,
    node_dict,
    dynamics_base_dir,
    node_types_df,
    neuron_class,
    sim_config,
):
    for i, model_name in enumerate(all_model_names):

        dynamics_params_renamed, num_neurons = get_dynamics_params(
            node_types_df, dynamics_base_dir, sim_config, node_dict, model_name
        )

        params = {k: dynamics_params_renamed[k] for k in neuron_class.get_param_names()}
        init = {
            k: dynamics_params_renamed[k]
            for k in ["V", "refractory_countdown", "ASC_1", "ASC_2"]
        }

        pop_dict[model_name] = model.add_neuron_population(
            pop_name=model_name,
            num_neurons=num_neurons,
            neuron=neuron_class,
            param_space=params,
            var_space=init,
        )

        # Assign extra global parameter values
        for k in pop_dict[model_name].extra_global_params.keys():
            pop_dict[model_name].set_extra_global_param(k, dynamics_params_renamed[k])

        print("{} population added to model.".format(model_name))
    return pop_dict


def construct_synapses(
    model,
    syn_dict,
    pop1,
    pop2,
    dynamics_base_dir,
    edges,
    edge_type_id,
    syn_df,
    sim_config,
    dynamics_params,
):
    # **TODO** all of this could be done per edge type rather than per synapse pop
    # Open dynamics file used by this edge type
    synaptic_dynamics_file = syn_df[syn_df["edge_type_id"] == edge_type_id][
        "dynamics_params"
    ].iloc[0]
    
    synaptic_dynamics_path = Path(dynamics_base_dir, synaptic_dynamics_file)
    with open(synaptic_dynamics_path) as f:
        synaptic_dynamics_params = json.load(f)
        
    # Get delay and weight specific to the edge_type_id
    delay_steps = round(
        syn_df[syn_df["edge_type_id"] == edge_type_id]["delay"].iloc[0]
        / sim_config["run"]["dt"]
    )  # delay (ms) -> delay (steps)
    weight = (
        syn_df[syn_df["edge_type_id"] == edge_type_id]["syn_weight"].iloc[0]
        / 1e3
    )  # nS -> uS; multiply by number of synapses
    
    tau_syn = dynamics_params["tau_syn"][synaptic_dynamics_params["receptor_type"] - 1]
    
    s_ini = {"g": weight}
    psc_Alpha_params = {"tau": tau_syn}
    psc_Alpha_init = {"x": 0.0}

    synapse_group_name = f"{pop1}_{pop2}_{edge_type_id}"

    syn_dict[synapse_group_name] = model.add_synapse_population(
        pop_name=synapse_group_name,
        matrix_type="SPARSE_GLOBALG_INDIVIDUAL_PSM",
        delay_steps=delay_steps,
        source=pop1,
        target=pop2,
        w_update_model="StaticPulse",
        wu_param_space={},
        wu_var_space=s_ini,
        wu_pre_var_space={},
        wu_post_var_space={},
        postsyn_model=psc_Alpha,
        ps_param_space=psc_Alpha_params,
        ps_var_space=psc_Alpha_init,
    )

    # **TODO** no reason not to have edges in this order and as sensible numpy type already
    edges = list(zip(*edges))
    syn_dict[synapse_group_name].set_sparse_connections(
        np.asarray(edges[0], dtype=np.int64), np.asarray(edges[1], dtype=np.int64)
    )
    print(
        f"Synapses added for {pop1} -> {pop2} with edge type id={edge_type_id}"
    )

    return syn_dict
