# Replicate point_450glifs example with GeNN
import numpy as np
import pandas as pd
import time
import json
from collections import defaultdict
from pathlib import Path
from sonata.circuit import File
from sonata.reports.spike_trains import SpikeTrains
import pygenn
import matplotlib.pyplot as plt
from utilities import (
    GLIF3,
    get_dynamics_params,
    spikes_list_to_start_end_times,
    psc_Alpha,
    construct_populations,
    construct_synapses,
)

CELL_DYNAMICS_BASE_DIR = Path("./point_components/cell_models")
SYNAPSE_DYNAMICS_BASE_DIR = Path("./point_components/synaptic_models")
SIM_CONFIG_PATH = Path("point_450glifs/config.simulation.json")
LGN_V1_EDGE_CSV = Path("./point_450glifs/network/lgn_v1_edge_types.csv")
V1_EDGE_CSV = Path("./point_450glifs/network/v1_v1_edge_types.csv")
LGN_SPIKES_PATH = Path("./point_450glifs/inputs/lgn_spikes.h5")
LGN_NODE_DIR = Path("./point_450glifs/network/lgn_node_types.csv")
V1_NODE_CSV = Path("./point_450glifs/network/v1_node_types.csv")
V1_ID_CONVERSION_FILENAME = Path(".", "pkl_data", "v1_edge_df.pkl")
LGN_ID_CONVERSION_FILENAME = Path(".", "pkl_data", "lgn_edge_df.pkl")
BKG_V1_EDGE_CSV = Path("./BKG_test.csv")
BKG_ID_CONVERSION_FILENAME = Path(".", "pkl_data", "bkg_edge_df.pkl")
NUM_RECORDING_TIMESTEPS = 10000
num_steps = 3000000


v1_net = File(
    data_files=[
        "point_450glifs/network/v1_nodes.h5",
        "point_450glifs/network/v1_v1_edges.h5",
    ],
    data_type_files=[
        "point_450glifs/network/v1_node_types.csv",
        "point_450glifs/network/v1_v1_edge_types.csv",
    ],
)

lgn_net = File(
    data_files=[
        "point_450glifs/network/lgn_nodes.h5",
        "point_450glifs/network/lgn_v1_edges.h5",
    ],
    data_type_files=[
        "point_450glifs/network/lgn_node_types.csv",
        "point_450glifs/network/lgn_v1_edge_types.csv",
    ],
)

print("Contains nodes: {}".format(v1_net.has_nodes))
print("Contains edges: {}".format(v1_net.has_edges))
print("Contains nodes: {}".format(lgn_net.has_nodes))
print("Contains edges: {}".format(lgn_net.has_edges))


### Create base model ###
with open(SIM_CONFIG_PATH) as f:
    sim_config = json.load(f)
model = pygenn.genn_model.GeNNModel(backend="SingleThreadedCPU")
model.dT = sim_config["run"]["dt"]

### Construct v1 neuron populations ###
v1_node_types_df = pd.read_csv(V1_NODE_CSV, sep=" ")
v1_nodes = v1_net.nodes["v1"]
v1_model_names = v1_node_types_df["model_name"]
v1_node_dict = {m : [n["node_id"] for n in v1_nodes.filter(dynamics_params=d)]
                for m, d in zip(v1_model_names,
                                v1_node_types_df["dynamics_params"])}

# Add populations
pop_dict = {}
pop_dict = construct_populations(
    model,
    pop_dict,
    all_model_names=v1_model_names,
    node_dict=v1_node_dict,
    dynamics_base_dir=CELL_DYNAMICS_BASE_DIR,
    node_types_df=v1_node_types_df,
    neuron_class=GLIF3,
    sim_config=sim_config,
)

# Enable spike recording
for k in pop_dict.keys():
    pop_dict[k].spike_recording_enabled = True

### Construct LGN neuron populations ###
lgn_node_types_df = pd.read_csv(LGN_NODE_DIR, sep=" ")
lgn_nodes = lgn_net.nodes["lgn"]
lgn_model_names = lgn_node_types_df["model_type"].to_list()
lgn_node_dict = {m: [n["node_id"] for n in lgn_nodes.filter(model_type=m)]
                 for m in lgn_model_names}
assert len(lgn_node_dict) == 1
assert "virtual" in lgn_node_dict

# Read LGN spike times
spikes = SpikeTrains.from_sonata(LGN_SPIKES_PATH)
spikes_df = spikes.to_dataframe()

# Add population
for name, nodes in lgn_node_dict.items():
    # Get list of spike times associated with node
    spikes_list = [spikes_df[spikes_df["node_ids"] == n]["timestamps"].to_numpy()
                   for n in nodes]

    # Convert to GeNN format
    (start_spike, end_spike, spike_times) =\
        spikes_list_to_start_end_times(spikes_list)

    pop_dict[name] = model.add_neuron_population(
        name,
        len(nodes),
        "SpikeSourceArray",
        {},
        {"startSpike": start_spike, "endSpike": end_spike},
    )

    pop_dict[name].set_extra_global_param("spikeTimes", spike_times)

### Construct BKG neuron population ###
BKG_name = "BKG"
BKG_params = {"rate": 1000}  # 1kHz
BKG_var = {"timeStepToSpike": 0}
pop_dict[BKG_name] = model.add_neuron_population(
    BKG_name,
    num_neurons=1,
    neuron="PoissonNew",
    param_space=BKG_params,
    var_space=BKG_var,
)

### Construct v1 to v1 synapses ###

# First create a list mapping the NEST node_id to the GeNN node_id. NEST numbers the neurons from 0 to num_neurons, whereas GeNN numbers neurons 0 to num_neurons_per_population. This matters when assigning synapses.
v1_node_to_pop_idx = [None] * len(v1_nodes)
for name, nodes in v1_node_dict.items():
    for n in nodes:
        v1_node_to_pop_idx[n] = (name, n - nodes[0])

# Loop through edges
# **TODO** put in function
v1_edges_dict = defaultdict(list)
for e in v1_net.edges["v1_to_v1"]:
    # Convert source and target node IDs to GeNN model and neuron index
    genn_source = v1_node_to_pop_idx[e.source_node_id]
    genn_trg = v1_node_to_pop_idx[e.target_node_id]

    # Add specified number of copies of source, target pair to dictionary
    v1_edges_dict[(genn_source[0], genn_trg[0], e.edge_type_id)].extend(
        [(genn_source[1], genn_trg[1])]* e["nsyns"])

# Read edge types
v1_syn_df = pd.read_csv(V1_EDGE_CSV, sep=" ")

# Loop through all the edges associated with each pair of populations/edge type
syn_dict = {}
for (pop_src, pop_trg, edge_type_id), edges in v1_edges_dict.items():
    # Get dynamics of target neuron (only used to get tau syn)
    dynamics_params, _ = get_dynamics_params(
        node_types_df=v1_node_types_df,
        dynamics_base_dir=CELL_DYNAMICS_BASE_DIR,
        sim_config=sim_config,
        node_dict=v1_node_dict,
        model_name=pop_trg,
    )

    syn_dict = construct_synapses(
        model=model,
        syn_dict=syn_dict,
        pop1=pop_src,
        pop2=pop_trg,
        dynamics_base_dir=SYNAPSE_DYNAMICS_BASE_DIR,
        edge_type_id=edge_type_id,
        edges=edges,
        syn_df=v1_syn_df,
        sim_config=sim_config,
        dynamics_params=dynamics_params,
    )

### Construct LGN to v1 synapses ###
# First create a list that maps the NEST node_id to the GeNN node_id. NEST numbers the neurons from 0 to num_neurons, whereas GeNN numbers neurons 0 to num_neurons_per_population. This matters when assigning synapses.
lgn_node_to_pop_idx = [None] * len(lgn_nodes)
for name, nodes in lgn_node_dict.items():
    for n in nodes:
        lgn_node_to_pop_idx[n] = (name, n - nodes[0])

# Loop through edges
# **TODO** put in function
lgn_edges_dict = defaultdict(list)
for e in lgn_net.edges["lgn_to_v1"].get_group(0):
    # Convert source and target node IDs to GeNN model and neuron index
    genn_source = lgn_node_to_pop_idx[e.source_node_id]
    genn_trg = v1_node_to_pop_idx[e.target_node_id]

    # Add specified number of copies of source, target pair to dictionary
    lgn_edges_dict[(genn_source[0], genn_trg[0], e.edge_type_id)].extend(
        [(genn_source[1], genn_trg[1])]* e["nsyns"])

# Read edge types
lgn_syn_df = pd.read_csv(LGN_V1_EDGE_CSV, sep=" ")

# Loop through all the edges associated with each pair of populations/edge type
for (pop_src, pop_trg, edge_type_id), edges in lgn_edges_dict.items():
    # Get dynamics of target neuron (only used to get tau syn)
    dynamics_params, _ = get_dynamics_params(
        node_types_df=v1_node_types_df,
        dynamics_base_dir=CELL_DYNAMICS_BASE_DIR,
        sim_config=sim_config,
        node_dict=v1_node_dict,
        model_name=pop_trg,  # Pop2 is target, used for dynamics_params (tau)
    )
    syn_dict = construct_synapses(
        model=model,
        syn_dict=syn_dict,
        pop1=pop_src,
        pop2=pop_trg,
        dynamics_base_dir=SYNAPSE_DYNAMICS_BASE_DIR,
        edge_type_id=edge_type_id,
        edges=edges,
        syn_df=lgn_syn_df,
        sim_config=sim_config,
        dynamics_params=dynamics_params,
    )

### Construct BKG to v1 synapses ###
"""
# Test BKG working with connection to all v1 with same weights
# Get delay and weight specific to the edge_type_id
nsyns = 21
weight = 0.192834123607 / 1e3 * nsyns  # nS -> uS; multiply by number of synapses
s_ini = {"g": weight}
psc_Alpha_params = {"tau": dynamics_params["tau_syn"][0]}  # TODO: Always 0th port?
psc_Alpha_init = {"x": 0.0}
pop1 = BKG_name
for pop2 in v1_model_names:
    synapse_group_name = pop1 + "_to_" + pop2 + "_nsyns_" + str(nsyns)
    syn_dict[synapse_group_name] = model.add_synapse_population(
        pop_name=synapse_group_name,
        matrix_type="SPARSE_GLOBALG_INDIVIDUAL_PSM",
        delay_steps=0,
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

    t_list = [i for i in range(pop_dict[pop2].size)]
    s_list = [0 for i in t_list]
    syn_dict[synapse_group_name].set_sparse_connections(
        np.array(s_list), np.array(t_list)
    )
    print(f"Synapses added for {pop1} -> {pop2}")
"""
### This commented out section is hard to run on this smaller dataset of 450 neurons because there is no bkg_nodes.h5 file for this dataset. Trying to copy the one from the Billeh dataset doesn't work, as it uses populations of neurons with different names.
# bkg_node_to_pop_idx = {0: [BKG_name, 0]}
# bkg_edges = bkg_net.edges["bkg_to_v1"].get_group(0)
# bkg_edge_df = construct_id_conversion_df(
#     edges=bkg_edges,
#     all_model_names=v1_model_names,
#     source_node_to_pop_idx_dict=bkg_node_to_pop_idx,
#     target_node_to_pop_idx_dict=v1_node_to_pop_idx,
#     filename=BKG_ID_CONVERSION_FILENAME,
# )

# pop1 = BKG_name
# bkg_syn_df = pd.read_csv(BKG_V1_EDGE_CSV, sep=" ")
# bkg_edge_type_ids = bkg_syn_df["edge_type_id"].tolist()
# bkg_all_nsyns = bkg_edge_df["nsyns"].unique()
# bkg_all_nsyns.sort()

# for pop2 in v1_model_names:

#     # Dynamics for v1, since this is the target
#     dynamics_params, _ = get_dynamics_params(
#         node_types_df=v1_node_types_df,
#         dynamics_base_dir=DYNAMICS_BASE_DIR,
#         sim_config=sim_config,
#         node_dict=v1_node_dict,
#         model_name=pop2,  # Pop2 is target, used for dynamics_params (tau)
#     )
#     syn_dict = construct_synapses(
#         model=model,
#         syn_dict=syn_dict,
#         pop1=pop1,
#         pop2=pop2,
#         all_edge_type_ids=bkg_edge_type_ids,
#         all_nsyns=bkg_all_nsyns,
#         edge_df=bkg_syn_df,
#         syn_df=bkg_syn_df,
#         sim_config=sim_config,
#         dynamics_params=dynamics_params,
#     )


### Run simulation ###
model.build()
model.load(
    num_recording_timesteps=NUM_RECORDING_TIMESTEPS
)  # TODO: How big to calculate for GPU size?

# Construct data for spike times
spike_data = {}
for model_name, nodes in v1_node_dict.items():
    spike_data[model_name] = [[] for _ in range(len(nodes))]

for i in range(num_steps):
    model.step_time()

    # Only collect full BUFFER
    if i % NUM_RECORDING_TIMESTEPS == 0 and i != 0:
        # Record spikes
        print(i)
        model.pull_recording_buffers_from_device()
        for model_name in v1_model_names:
            pop = pop_dict[model_name]
            spk_times, spk_ids = pop.spike_recording_data
            for st, si in zip(spk_times, spk_ids):
                spike_data[model_name][si].append(st)

# Create list of spike times associated with BMTK node IDs
spike_data_BMTK_ids = [spike_data[n][i]
                       for n, i in v1_node_to_pop_idx]

# Create reverse lookup from (model name, pop id) tuple back to node ID
# **THINK** wasteful - all you really need is starting ids of each population
v1_node_to_pop_idx_inv = {}
for node_id, pop_name_id in enumerate(v1_node_to_pop_idx):
    v1_node_to_pop_idx_inv[str(pop_name_id)] = node_id

# Plot firing rates
fig, axs = plt.subplots(1, 1)
period_length = (num_steps * sim_config["run"]["dt"]) / 1000.0  # ms
for model_name in sorted(v1_model_names):
    ids = [v1_node_to_pop_idx_inv[str((model_name, id))]
           for id, _ in enumerate(spike_data[model_name])]
    firing_rates = [len(t) / period_length
                    for t in spike_data[model_name]]
    axs.plot(ids, firing_rates, "o", label=model_name)

axs.set_ylabel("Firing Rate (hz)")
axs.set_xlabel("node_id")
axs.legend()
plt.show()

print("Simulation complete.")
