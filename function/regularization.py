import torch

def layer_wise_regularization(spikes_tensor, target_spike_proba, time_steps):
    target_nb_spikes = spikes_tensor.size(-1)*target_spike_proba*time_steps
    nb_spike_at_layer = spikes_tensor.sum(-1)
    diff = torch.relu(nb_spike_at_layer - target_nb_spikes)**2
    with torch.no_grad():
        spike_proba = nb_spike_at_layer / (time_steps*spikes_tensor.size(-1))
        spike_proba = torch.mean(spike_proba)
    return torch.mean(diff / time_steps), spike_proba
