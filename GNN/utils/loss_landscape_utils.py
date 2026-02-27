import torch
from torch import autocast, nn, max, min, no_grad, cat, save, load, distributed, tensor, is_tensor
import torch.nn as nn
from contextlib import contextmanager
import tqdm

def make_random_direction(model):
    direction = {}
    for name, param in model.named_parameters():
        # Find the module that owns this parameter
        module_names = name.split(".")
        submodule = model
        for mn in module_names[:-1]:
            if mn.isdigit():
                submodule = submodule[int(mn)]  # Sequential index
            else:
                submodule = getattr(submodule, mn)

        # If that submodule is LayerNorm, skip
        if isinstance(submodule, torch.nn.LayerNorm):
            direction[name] = torch.zeros_like(param)
        else:
            direction[name] = torch.randn_like(param)
    return direction

def normalize_direction(direction, model):
    eps = 1e-12
    normed = {}
    for name, param in model.named_parameters():
        d = direction[name]

        # Skip norms
        if "norm" in name.lower():
            normed[name] = torch.zeros_like(param)
            continue

        # Linear weights
        if param.ndim == 2:  # [out_features, in_features]
            d_flat = d.clone()
            p_flat = param.detach()

            d_norms = d_flat.norm(dim=1, keepdim=True).clamp_min(eps)
            p_norms = p_flat.norm(dim=1, keepdim=True)

            scaled = d_flat / d_norms * p_norms
            normed[name] = scaled

        # Biases (1D)
        elif param.ndim == 1:
            d_norm = d.norm().clamp_min(eps)
            p_norm = param.detach().norm()
            scaled = d / d_norm * p_norm
            normed[name] = scaled

        else:
            # fallback (e.g., embeddings, other shapes)
            normed[name] = torch.zeros_like(param)
    return normed




# ---------- helper context: temporarily perturb weights ----------
@contextmanager
def perturb_weights(model, delta, eta, alpha, beta, parallel, ddp):
    """Temporarily apply αδ+βη to model parameters, then restore."""
    if parallel:
        m = model.module
    else:
        m = model

    if ddp:
        m = m.module
    snapshot = {n: p.clone() for n, p in m.named_parameters()}
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.add_(alpha * delta[n] + beta * eta[n])
    try:
        yield
    finally:
        with torch.no_grad():
            for n, p in m.named_parameters():
                p.copy_(snapshot[n])


@torch.no_grad()
def apply_perturbation(model, delta, eta, alpha, beta, parallel=False, ddp=False):
    """Apply αδ+βη to model parameters in place."""
    if parallel:
        m = model.module
    else:
        m = model
    if ddp:
        m = m.module

    for n, p in m.named_parameters():
        p.copy_(p + alpha * delta[n] + beta * eta[n])
# ---------- evaluation for one perturbed model ----------
def eval_perturbed_model(
    model,
    alpha, 
    beta,  
    data_loader, 
    device,
    loss_function,
    parallel=False,
    ddp=False,
    env=None,
     
    ):
    
    amp_device = "cpu" if device == "cpu" else "cuda"
    amp = False
    model.eval()
    total_loss = 0.0
    total_samples = 0

    
    tqdm_itr = tqdm.tqdm(data_loader, position=1, desc=f"Eval α={alpha:.2f}, β={beta:.2f}", leave=False)
        
    for i, data_list in enumerate(tqdm_itr):
        with autocast(amp_device, enabled=amp):
            if not parallel:
                data_list = data_list.to(device)

            with no_grad():
                predicted = model(data_list)
                if parallel:
                    y = cat([data.y for data in data_list]).to(model.device)
                    m = model.module
                else:
                    y = data_list.y
                    m = model

                if ddp:
                    m = m.module

                assert hasattr(m, "_output_normalizer")
                y = m._output_normalizer(y, accumulate=m.training)

                # implement "get_non_source_data_mask"
                dataset = data_loader.dataset
                mask = dataset.get_non_source_data_mask(data_list)
                
                # compute loss
                loss = loss_function(predicted[mask], y[mask])
                #print('loss', loss)
        

            total_loss += loss.item()
            
    if ddp:
        
            
        t = tensor(total_loss, device=device)
        distributed.all_reduce(t)   # does sum reduction by default...
        total_loss = t.item() / env.world_size
            
            
    total_loss = total_loss / len(data_loader)

    
    return total_loss