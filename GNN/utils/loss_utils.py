from modulus.sym.eq.pde import PDE
from modulus.sym.eq.phy_informer import PhysicsInformer
from modulus.sym.eq.spatial_grads.spatial_grads import compute_connectivity_tensor
from sympy import Function, Number, Symbol
import torch.nn as nn
import torch
from torch import tensor, cat, int64, is_tensor
from torch.nn.functional import one_hot, pad
from numpy import ones, squeeze, diff, tile
from torch_geometric.data import Data
import numpy as np
import h5py
import torch.distributed as dist

def create_hdf5(hf : h5py._hl.files.File , dict : dict):
    '''
    write lists into a hdf5 dataset
    '''
    keys = dict.keys()

    for k in keys:
        hf.create_dataset(k, data = dict[k])

    hf.close()

class Euler(PDE):
    """Euler equation for Sedov Taylor explosion"""

    def __init__(self, gamma, dim=3):
        # set params
        self.dim = dim

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z}
        if self.dim == 2:
            input_variables.pop("z")

        # velocity componets
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        if self.dim == 3:
            w = Function("w")(*input_variables)
        else:
            w = Number(0)

        # acceleration components
        u_t = Function("u_t")(*input_variables)
        v_t = Function("v_t")(*input_variables)
        if self.dim == 3:
            w_t = Function("w_t")(*input_variables)
        else:
            w_t = Number(0)

        # pressure
        p = Function("p")(*input_variables)

        # density
        rho = Function("rho")(*input_variables)

        #dynamics
        p_t = Function("p_t")(*input_variables)
        rho_t = Function("rho_t")(*input_variables)

        #ideal gas gamma
        if isinstance(gamma, str):
            gamma = Function(gamma)(*input_variables)
        elif isinstance(gamma, (float, int)):
            gamma = Number(gamma)

        # set equations
        epsilon = 1e-6
        self.equations = {}
        self.equations["continuity"] = (
            rho_t + rho * (u.diff(x) + v.diff(y))
        )
        self.equations["momentum_x"] = (
            u_t * rho + (p.diff(x))
        )
        self.equations["momentum_y"] = (
            v_t * rho + (p.diff(y))
        )
        self.equations["ideal_gas"] = (
            p_t * rho ** (-gamma) + p * (-gamma) * rho ** (-gamma - 1) * rho_t
        )
        #self.equations["continuity"] = (
        #    rho_t + u * rho.diff(x) + v * rho.diff(y) + rho * (u.diff(x) + v.diff(y))
        #)
        #self.equations["momentum_x"] = (
        #    u_t + u * u.diff(x) + v * u.diff(y) + (p.diff(x))/rho
        #)
        #self.equations["momentum_y"] = (
        #    v_t + u * v.diff(x) + v * v.diff(y) + (p.diff(y))/rho
        #)
        #self.equations["ideal_gas"] = (
        #    p_t * rho ** (-gamma) + p * (-gamma) * rho ** (-gamma - 1) * rho_t + u * (p * rho ** (-gamma)).diff(x) + v * (p * rho ** (-gamma)).diff(y) + w * (p * rho ** (-gamma)).diff(z)
        #)

        #self.equations["ideal_gas"] = (
        #    p_t * abs(rho ** (-gamma)) + p * (-gamma) * abs(rho ** (-gamma - 1)) * rho_t + u * (p * abs(rho ** (-gamma))).diff(x) + v * (p * abs(rho ** (-gamma))).diff(y) + w * (p * abs(rho ** (-gamma))).diff(z)
        #)

        # Not working: rho**1.4
        #              .diff() using least square method
        # Note : found the cause of backpropagation instability --- spacial derivative with least square method leads to gradien explosion in back propagation. when there is no spacial derivative, it is fine.
        # Fixed by using finite difference method instead

class Euler_compress(PDE):
    """Euler equation for Sedov Taylor explosion"""

    def __init__(self, gamma, dim=3):
        # set params
        self.dim = dim

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z}
        if self.dim == 2:
            input_variables.pop("z")

        # velocity componets
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        if self.dim == 3:
            w = Function("w")(*input_variables)
        else:
            w = Number(0)

        # acceleration components
        u_t = Function("u_t")(*input_variables)
        v_t = Function("v_t")(*input_variables)
        if self.dim == 3:
            w_t = Function("w_t")(*input_variables)
        else:
            w_t = Number(0)

        # pressure
        p = Function("p")(*input_variables)

        # density
        rho = Function("rho")(*input_variables)

        #dynamics
        p_t = Function("p_t")(*input_variables)
        rho_t = Function("rho_t")(*input_variables)
        rho_u_t = Function("rho_u_t")(*input_variables)
        rho_v_t = Function("rho_v_t")(*input_variables)

        #ideal gas gamma
        if isinstance(gamma, str):
            gamma = Function(gamma)(*input_variables)
        elif isinstance(gamma, (float, int)):
            gamma = Number(gamma)

        # set equations
        epsilon = 1e-6
        self.equations = {}
        self.equations["continuity"] = (
            rho_t + rho * (u.diff(x) + v.diff(y))
        )
        self.equations["momentum_x"] = (
            rho_u_t + p.diff(x) + (rho * u) * (u.diff(x)) + (rho * u * v).diff(y) 
        )
        self.equations["momentum_y"] = (
            rho_v_t + p.diff(y) + (rho * v) * (v.diff(y)) + (rho * u * v).diff(x) 
        )
        self.equations["ideal_gas"] = (
            p_t + gamma * p * ( u.diff(x) + v.diff(y))
        )
        

class Euler_rel(PDE):
    """Euler equation for Sedov Taylor explosion"""

    def __init__(self, gamma, dim=3):
        # set params
        self.dim = dim

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z}
        if self.dim == 2:
            input_variables.pop("z")

        # velocity componets
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        if self.dim == 3:
            w = Function("w")(*input_variables)
        else:
            w = Number(0)
        g = 1/(1-(u**2 + v**2))**0.5
        # acceleration components
        

        # pressure
        p = Function("p")(*input_variables)

        # density
        rho = Function("rho")(*input_variables)
        
        h = 1 + (p * gamma/(gamma-1))/rho
        #dynamics
        p_t = Function("p_t")(*input_variables)
        rho_t = Function("rho_t")(*input_variables)
        mom_u_t = Function("mom_u_t")(*input_variables)
        mom_v_t = Function("mom_v_t")(*input_variables)

        #ideal gas gamma
        if isinstance(gamma, str):
            gamma = Function(gamma)(*input_variables)
        elif isinstance(gamma, (float, int)):
            gamma = Number(gamma)

        # set equations
        epsilon = 1e-6
        
        #self.equations["continuity"] = (
        #    rho_t + g * rho * (u.diff(x) + v.diff(y))
        #)
        #self.equations["momentum_x"] = (
        #    mom_u_t + p.diff(x) + (g**2 * h * rho * u) * (u.diff(x)) + (g**2 * h * rho * u * v).diff(y) 
        #)
        #self.equations["momentum_y"] = (
        #    mom_v_t + p.diff(y) + (g**2 * h * rho * v) * (v.diff(y)) + (g**2 * h * rho * u * v).diff(x) 
        #)
        #self.equations["ideal_gas"] = (
        #    p_t + gamma * p * ( u.diff(x) + v.diff(y))
            
        #)
        self.equations = {}
        self.equations["continuity"] = (
            rho_t + 1 * rho * (u.diff(x) + v.diff(y))
        )
        self.equations["momentum_x"] = (
            mom_u_t + p.diff(x) + (  h * rho * u) * (u.diff(x)) + ( h * rho * u * v).diff(y) 
        )
        self.equations["momentum_y"] = (
            mom_v_t + p.diff(y) + ( h * rho * v) * (v.diff(y)) + ( h * rho * u * v).diff(x) 
        )
        self.equations["ideal_gas"] = (
            p_t + gamma * p * ( u.diff(x) + v.diff(y))
            
        )




class PINNLoss(nn.Module):
    def __init__(self, device, m):
        super(PINNLoss, self).__init__()
        self.device = device
        #self.node_pde = Euler(gamma = 1.4, dim=2)
        
        grad_method = m.get_grad_method()
        forward_method = m.get_pinn_forward_method()
        fd_dx = m.get_fd_dx()
        PDE = m.get_PDE()
        print('PDE function is', PDE)
        if PDE == 'Euler_compress':
            self.node_pde = Euler_compress(gamma = 1.4, dim=2)
        elif PDE == 'Euler_rel':
            self.node_pde = Euler_rel(gamma = 1.4, dim=2)
        
        self.dt = m.get_dt()
        self.xsize = m.get_xsize()
        self.ysize = m.get_ysize()
        self.full_xsize = m.get_full_xsize()
        self.full_ysize = m.get_full_ysize()
        self.phy_informer = PhysicsInformer(
            required_outputs=["momentum_x", "momentum_y", "continuity", "ideal_gas"],
            equations=self.node_pde,
            grad_method=grad_method,
            fd_dx =fd_dx, # Hard coded for Sedov dataset, not 1, use real value 0.0156
            device=self.device,
            compute_connectivity=False,
        )
        print('grad method', self.phy_informer.grad_method)
        print('required outputs', self.phy_informer.required_outputs)
        print('required inputs', self.phy_informer.required_inputs)
        print('forward method', forward_method)
        forward_methods = {
        "no_pinn": self._forward,
        "boundary_mask": self.boundary_forward, 
        "shock_mask":self.shock_forward, 
        "boundary_mask_full_pinn": self.double_forward, 
        "boundary_mask_GT_res": self.GT_res_forward,
        "boundary_mask_GT_res_backward": self.GT_res_backward,
        "boundary_mask_GT_res_backward_rel": self.GT_res_backward_rel,
        "boundary_mask_GT_res_backward_weighted_data": self.GT_res_backward_weighted_data,
        "boundary_mask_GT_res_backward_weighted_data_10p": self.GT_res_backward_weighted_data_10p,
        "boundary_mask_full_pinn_GT_res": self.double_GT_res_forward,
        "data_single_parameter": self.data_single_parameter,
        #"shock_mask_full_pinn": self.shock_fp_forward, 
        }
        assert forward_method in forward_methods
        self.forward = forward_methods[forward_method]
        print(f'using {forward_method} as forward method for PINNLoss')
        self.input_type = m.get_input_type()
        self.output_type = m.get_output_type()
        output_type = self.output_type
        print(f'output type {output_type}')
        #assert self.output_type == "velocity & density & pressure"
    
    def _forward(self, data_list, model, parallel, non_source_mask, train):
    
        
        pred = model(data_list)
        if model.normalize_output:
            mgn_output = model._output_normalizer.inverse(pred)
        else:
            mgn_output = pred


        batch_num = len(data_list.ptr)-1

        if parallel:
            #y is ground truth data
            y = cat([data.y for data in data_list]).to(pred.device)
            x = cat([data.x for data in data_list]).to(pred.device)

        else:
            y = data_list.y
            x = data_list.x
            pres_hist = data_list.pressure

        assert hasattr(model, "_output_normalizer")
        y_norm = model._output_normalizer(y, accumulate=model.training)
        y_norm = y_norm[:, :4]
        loss_fn = nn.MSELoss()
        loss_data = loss_fn(pred[non_source_mask], y_norm[non_source_mask])
        
        loss_ideal = torch.tensor([0]).to(self.device)
        loss_cont = torch.tensor([0]).to(self.device)
        loss_mx = torch.tensor([0]).to(self.device)
        loss_my = torch.tensor([0]).to(self.device)
        loss = {
                'ideal_gas':loss_ideal,
                'continuity': loss_cont,
                'momentum_x': loss_mx, 
                'momentum_y': loss_my,
                'data': loss_data
                }
        return loss
        
            
   
    def boundary_forward(self, data_list, model, parallel, non_source_mask, train):
    
        if train:
            xsize = self.xsize
            ysize = self.ysize
        elif not train:
            xsize = self.full_xsize
            ysize = self.full_ysize
        # Note: pred must be denormalised before using
        pred = model(data_list)
        if model.normalize_output:
            mgn_output = model._output_normalizer.inverse(pred)
        else:
            mgn_output = pred


        batch_num = len(data_list.ptr)-1

        if parallel:
            #y is ground truth data
            y = cat([data.y for data in data_list]).to(pred.device)
            x = cat([data.x for data in data_list]).to(pred.device)

        else:
            y = data_list.y
            x = data_list.x
            pres_hist = data_list.pressure

        assert hasattr(model, "_output_normalizer")
        y_norm = model._output_normalizer(y, accumulate=model.training)
        y_norm = y_norm[:, :4]
        loss_fn = nn.MSELoss()
        loss_data = loss_fn(pred[non_source_mask], y_norm[non_source_mask])
        

        
        # Note: the model output is du/dt, dv/dt, drho/dt, p
        # Now d/dt is just u(t+Δt)-u(t), need to divide by Δt, hard code Δt = 0.005
        #Later found dt=1, dx = 1 fits PDE better ### No, it just scales 
        dt = self.dt
        
        results_int = self.phy_informer.forward(
            {
                 "u": x[:, 0:1].reshape(batch_num, 1, xsize, ysize),
                "v": x[:, 1:2].reshape(batch_num, 1, xsize, ysize),
                "u_t": (mgn_output[:, 0:1]/dt).reshape(batch_num, 1, xsize, ysize),
                "v_t": (mgn_output[:, 1:2]/dt).reshape(batch_num, 1, xsize, ysize),
                "rho_t": (mgn_output[:, 2:3]/dt).reshape(batch_num, 1, xsize, ysize),
                "p_t": ((mgn_output[:, 3:4] - pres_hist)/dt).reshape(batch_num, 1, xsize, ysize),
                "rho": x[:, 2:3].reshape(batch_num, 1, xsize, ysize),
                "p": pres_hist.reshape(batch_num, 1, xsize, ysize)
            }
        )

        res_ideal, res_cont, res_mx, res_my=( 
                results_int["ideal_gas"],
                results_int["continuity"],
                results_int["momentum_x"], 
                results_int["momentum_y"], 
            )
            
        
        #self.safe_stats("res_ideal", res_ideal)
        #self.safe_stats("res_cont", res_cont)
        #self.safe_stats("res_mx", res_mx)
        #self.safe_stats("res_my", res_my)
       
        
        non_source_mask = non_source_mask.reshape(batch_num, 1, xsize, ysize)
        
        loss_ideal = torch.mean((res_ideal[non_source_mask])** 2 )
        loss_cont = torch.mean((res_cont[non_source_mask])** 2 )
        loss_mx = torch.mean((res_mx[non_source_mask])** 2 )
        loss_my = torch.mean((res_my[non_source_mask])** 2 )
        loss = {
                'ideal_gas':loss_ideal,
                'continuity': loss_cont,
                'momentum_x': loss_mx, 
                'momentum_y': loss_my,
                'data': loss_data
                }
        return loss
        
        
        
    def shock_forward(self, data_list, model, parallel, non_source_mask, train):
        if train:
            xsize = self.xsize
            ysize = self.ysize
        elif not train:
            xsize = self.full_xsize
            ysize = self.full_ysize
        # Note: pred must be denormalised before using
        pred = model(data_list)
        if model.normalize_output:
            mgn_output = model._output_normalizer.inverse(pred)
        else:
            mgn_output = pred


        batch_num = len(data_list.ptr)-1

        if parallel:
            #y is ground truth data
            y = cat([data.y for data in data_list]).to(predicted.device)
            x = cat([data.x for data in data_list]).to(pred.device)
        else:
            y = data_list.y
            x = data_list.x
            pres_hist = data_list.pressure

        assert hasattr(model, "_output_normalizer")
        y_norm = model._output_normalizer(y, accumulate=model.training)
        y_norm = y_norm[:, :4]
        loss_fn = nn.MSELoss()
        loss_data = loss_fn(pred[non_source_mask], y_norm[non_source_mask])
        

        one_hot = data_list.x[:, 3:]
        target_type = torch.tensor([1, 0, 0, 0], dtype=data_list.x.dtype).to(one_hot.device)

        # Compare and create the mask
        post_shock_mask = (one_hot == target_type).all(dim=1) 

        #print('post shock elements idx2', post_shock_mask.sum())

        post_shock_mask = post_shock_mask & non_source_mask
        #print('post shock elements sub', post_shock_mask.sum())
        post_shock_mask = post_shock_mask.reshape(batch_num, 1, xsize, ysize)
        # Note: the model output is du/dt, dv/dt, drho/dt, p
        # Now d/dt is just u(t+Δt)-u(t), need to divide by Δt, hard code Δt = 0.005
        #Later found dt=1, dx = 1 fits PDE better ### No, it just scales 
        dt = self.dt
            
        #back propagation tested, success
        results_int = self.phy_informer.forward(
            {
                 "u": x[:, 0:1].reshape(batch_num, 1, xsize, ysize),
                "v": x[:, 1:2].reshape(batch_num, 1, xsize, ysize),
                "u_t": (mgn_output[:, 0:1]/dt).reshape(batch_num, 1, xsize, ysize),
                "v_t": (mgn_output[:, 1:2]/dt).reshape(batch_num, 1, xsize, ysize),
                "rho_t": (mgn_output[:, 2:3]/dt).reshape(batch_num, 1, xsize, ysize),
                "p_t": ((mgn_output[:, 3:4] - pres_hist)/dt).reshape(batch_num, 1, xsize, ysize),
                "rho": x[:, 2:3].reshape(batch_num, 1, xsize, ysize),
                "p": pres_hist.reshape(batch_num, 1, xsize, ysize)
            }
        )



        res_ideal, res_cont, res_mx, res_my=( 
                results_int["ideal_gas"],
                results_int["continuity"],
                results_int["momentum_x"], 
                results_int["momentum_y"], 
            )
        
        loss_ideal = torch.mean((res_ideal[post_shock_mask])** 2 )
        loss_cont = torch.mean((res_cont[post_shock_mask])** 2 )
        loss_mx = torch.mean((res_mx[post_shock_mask])** 2 )
        loss_my = torch.mean((res_my[post_shock_mask])** 2 )
        loss = {
                'ideal_gas':loss_ideal,
                'continuity': loss_cont,
                'momentum_x': loss_mx, 
                'momentum_y': loss_my,
                'data': loss_data
                }
        return loss
        
    def double_forward(self,data_list, model, parallel, non_source_mask, train):
        torch.autograd.set_detect_anomaly(True)
        
        assert model.out_dim == 4
        assert data_list.x != None
        assert data_list.y != None
        assert self.input_type == "velocity & density"
        
        # Note: pred must be denormalised before using
        
        if train:
            xsize = self.xsize
            ysize = self.ysize
        elif not train:
            xsize = self.full_xsize
            ysize = self.full_ysize
        
        pred = model(data_list)
    
        # Note: pred must be denormalised before using in PDELoss
        if model.normalize_output:
            mgn_output = model._output_normalizer.inverse(pred)
        else:
            mgn_output = pred
        # to get f_2, first update y_t with f_1
        other_dim = data_list.x.shape[-1] - (model.out_dim-1)
      
        if model.normalize_output:  # f_1 = normalized dy/dt
            # update state (returns u(t+Δt), v(t+Δt), rho(t+Δt), p(t+Δt))
            # Next_statee is denormalised
            next_state = data_list.x + pad(mgn_output[:, :3], (0, other_dim))
            # apply MGN model
            # Note force_accum_off=True --- pred_2 is not normalised
            # pred_2 -- returns du(t+Δt)/dt, dv(t+Δt)/dt, drho(t+Δt)/dt, dp(t+Δt)/dt
            pred_2 = model(
                (
                next_state,
                data_list.edge_index,
                data_list.edge_attr,
                ),
                force_accum_off=True,
            )
        else:  # f_1 = dy/dt
            next_state = data_list.x + pad(pred, (0, other_dim))
            f_2 = self._forward(
                (
                next_state,
                data_list.edge_index,
                data_list.edge_attr,
                ),
                force_accum_off=True,
            )
        # Note: pred must be denormalised before using in PDELoss

        batch_num = len(data_list.ptr)-1

        if parallel:
            #y is ground truth data
            y = cat([data.y for data in data_list]).to(pred.device)

        else:
            y = data_list.y
            pres_hist = data_list.pressure

        assert hasattr(model, "_output_normalizer")
        y_norm = model._output_normalizer(y, accumulate=model.training)
        y_norm = y_norm[:, :4]
        loss_fn = nn.MSELoss()
        loss_data = loss_fn(pred[non_source_mask], y_norm[non_source_mask])
        
        
        # At time step t, the predicted solutions of the next time step t+Δt are separated
        # from the calculation graph and then flowed into the network to calculate solutions
        # for the step after next t + 2Δt.

        # Note: the model output is du/dt, dv/dt, drho/dt, p
        # Now d/dt is just u(t+Δt)-u(t), need to divide by Δt, hard code Δt = 0.005
        dt = self.dt

        results_int = self.phy_informer.forward(
            {
                "u": next_state[:, 0:1].reshape(batch_num, 1, xsize, ysize),
                "v": next_state[:, 1:2].reshape(batch_num, 1, xsize, ysize),
                "u_t": (pred_2[:, 0:1]/dt).reshape(batch_num, 1, xsize, ysize),
                "v_t": (pred_2[:, 1:2]/dt).reshape(batch_num, 1, xsize, ysize),
                "rho_t": (pred_2[:, 2:3]/dt).reshape(batch_num, 1, xsize, ysize),
                "p_t": (pred_2[:, 2:3]/dt).reshape(batch_num, 1, xsize, ysize),  # ASSUMING NOT calculating ideal_gas
                "rho": next_state[:, 2:3].reshape(batch_num, 1, xsize, ysize),
                "p": mgn_output[:, 3:4].reshape(batch_num, 1, xsize, ysize)
            }
        )


        res_ideal, res_cont, res_mx, res_my=( 
                results_int["ideal_gas"],  
                results_int["continuity"],
                results_int["momentum_x"], 
                results_int["momentum_y"], 
            )
        #if train:
        #    self.safe_stats("res_ideal", res_ideal)
        #    self.safe_stats("res_cont", res_cont)
        #    self.safe_stats("res_mx", res_mx)
        #    self.safe_stats("res_my", res_my)
        #else:
        #    pass
        
        non_source_mask = non_source_mask.reshape(batch_num, 1, xsize, ysize)
        
        loss_ideal = torch.mean((res_ideal[non_source_mask])** 2 )
        loss_cont = torch.mean((res_cont[non_source_mask])** 2 )
        loss_mx = torch.mean((res_mx[non_source_mask])** 2 )
        loss_my = torch.mean((res_my[non_source_mask])** 2 )
        loss = {
                'ideal_gas':loss_ideal,
                'continuity': loss_cont,
                'momentum_x': loss_mx, 
                'momentum_y': loss_my,
                'data': loss_data
                }
                
        return loss
        
    def safe_stats(self, name, tensor):
        if tensor is None:
            print(f"{name} is None")
            return
        print(f"{name}: min={tensor.min().item():.2e}, max={tensor.max().item():.2e}, mean={tensor.mean().item():.2e}, has_nan={torch.isnan(tensor).any().item()}")

    
    def double_GT_res_forward(self,data_list, model, parallel, non_source_mask, train):
        torch.autograd.set_detect_anomaly(True)
        
        assert model.out_dim == 4
        assert data_list.x != None
        assert data_list.y != None
        assert self.input_type == "velocity & density"
        
        # Note: pred must be denormalised before using
        
        if train:
            xsize = self.xsize
            ysize = self.ysize
        elif not train:
            xsize = self.full_xsize
            ysize = self.full_ysize
        
        pred = model(data_list)
    
        # Note: pred must be denormalised before using in PDELoss
        if model.normalize_output:
            mgn_output = model._output_normalizer.inverse(pred)
        else:
            mgn_output = pred
        # to get f_2, first update y_t with f_1
        other_dim = data_list.x.shape[-1] - (model.out_dim-1)
      
        if model.normalize_output:  # f_1 = normalized dy/dt
            # update state (returns u(t+Δt), v(t+Δt), rho(t+Δt), p(t+Δt))
            # Next_statee is denormalised
            next_state = data_list.x + pad(mgn_output[:, :3], (0, other_dim))
            # apply MGN model
            # Note force_accum_off=True --- pred_2 is not normalised
            # pred_2 -- returns du(t+Δt)/dt, dv(t+Δt)/dt, drho(t+Δt)/dt, dp(t+Δt)/dt
            pred_2 = model(
                (
                next_state,
                data_list.edge_index,
                data_list.edge_attr,
                ),
                force_accum_off=True,
            )
        else:  # f_1 = dy/dt
            next_state = data_list.x + pad(pred, (0, other_dim))
            f_2 = self._forward(
                (
                next_state,
                data_list.edge_index,
                data_list.edge_attr,
                ),
                force_accum_off=True,
            )
        # Note: pred must be denormalised before using in PDELoss

        batch_num = len(data_list.ptr)-1

        if parallel:
            #y is ground truth data
            y = cat([data.y for data in data_list]).to(pred.device)
            x = cat([data.x for data in data_list]).to(pred.device)

        else:
            y = data_list.y
            x = data_list.x
            pres_hist = data_list.pressure
            extra = data_list.extra

        assert hasattr(model, "_output_normalizer")
        y_norm = model._output_normalizer(y, accumulate=model.training)
        y_norm = y_norm[:, :4]
        loss_fn = nn.MSELoss()
        loss_data = loss_fn(pred[non_source_mask], y_norm[non_source_mask])
        
        
        # At time step t, the predicted solutions of the next time step t+Δt are separated
        # from the calculation graph and then flowed into the network to calculate solutions
        # for the step after next t + 2Δt.

        # Note: the model output is du/dt, dv/dt, drho/dt, p
        # Now d/dt is just u(t+Δt)-u(t), need to divide by Δt, hard code Δt = 0.005
        dt = self.dt
        
        print(extra.shape)
        
        results_int_GT = self.phy_informer.forward(
            {
                "u": extra[:, 3:4].reshape(batch_num, 1, xsize, ysize),
                "v": extra[:, 4:5].reshape(batch_num, 1, xsize, ysize),
                "u_t": (extra[:, 0:1]/dt).reshape(batch_num, 1, xsize, ysize),
                "v_t": (extra[:, 1:2]/dt).reshape(batch_num, 1, xsize, ysize),
                "rho_t": (extra[:, 2:3]/dt).reshape(batch_num, 1, xsize, ysize),
                "p_t": ((mgn_output[:, 3:4] - pres_hist)/dt).reshape(batch_num, 1, xsize, ysize),
                "rho": extra[:, 5:6].reshape(batch_num, 1, xsize, ysize),
                "p": (y[:, 3:4]).reshape(batch_num, 1, xsize, ysize)
            }
        )
        
        results_int = self.phy_informer.forward(
            {
                "u": next_state[:, 0:1].reshape(batch_num, 1, xsize, ysize),
                "v": next_state[:, 1:2].reshape(batch_num, 1, xsize, ysize),
                "u_t": (pred_2[:, 0:1]/dt).reshape(batch_num, 1, xsize, ysize),
                "v_t": (pred_2[:, 1:2]/dt).reshape(batch_num, 1, xsize, ysize),
                "rho_t": (pred_2[:, 2:3]/dt).reshape(batch_num, 1, xsize, ysize),
                "p_t": (pred_2[:, 2:3]/dt).reshape(batch_num, 1, xsize, ysize),  # ASSUMING NOT calculating ideal_gas
                "rho": next_state[:, 2:3].reshape(batch_num, 1, xsize, ysize),
                "p": mgn_output[:, 3:4].reshape(batch_num, 1, xsize, ysize)
            }
        )


        res_ideal, res_cont, res_mx, res_my=( 
                results_int["ideal_gas"] - results_int_GT["ideal_gas"],  #plaeceholder for ideal gas law equation
                results_int["continuity"] - results_int_GT["continuity"],
                results_int["momentum_x"] - results_int_GT["momentum_x"], 
                results_int["momentum_y"] - results_int_GT["momentum_y"], 
            )
        #if train:
        #    self.safe_stats("res_ideal", res_ideal)
        #    self.safe_stats("res_cont", res_cont)
        #    self.safe_stats("res_mx", res_mx)
        #    self.safe_stats("res_my", res_my)
        #else:
        #    pass
        
        non_source_mask = non_source_mask.reshape(batch_num, 1, xsize, ysize)
        
        loss_ideal = torch.mean((res_ideal[non_source_mask])** 2 )
        loss_cont = torch.mean((res_cont[non_source_mask])** 2 )
        loss_mx = torch.mean((res_mx[non_source_mask])** 2 )
        loss_my = torch.mean((res_my[non_source_mask])** 2 )
        loss = {
                'ideal_gas':loss_ideal,
                'continuity': loss_cont,
                'momentum_x': loss_mx, 
                'momentum_y': loss_my,
                'data': loss_data
                }
                
        return loss
        
        
        
    def GT_res_forward(self, data_list, model, parallel, non_source_mask, train):
    
        if train:
            xsize = self.xsize
            ysize = self.ysize
        elif not train:
            xsize = self.full_xsize
            ysize = self.full_ysize
        # Note: pred must be denormalised before using
        pred = model(data_list)
        if model.normalize_output:
            mgn_output = model._output_normalizer.inverse(pred)
        else:
            mgn_output = pred


        batch_num = len(data_list.ptr)-1

        if parallel:
            #y is ground truth data
            y = cat([data.y for data in data_list]).to(pred.device)
            x = cat([data.x for data in data_list]).to(pred.device)

        else:
            y = data_list.y
            x = data_list.x
            pres_hist = data_list.pressure
        
        assert hasattr(model, "_output_normalizer")
        y_norm = model._output_normalizer(y, accumulate=model.training)
        y_norm = y_norm[:, :4]
        loss_fn = nn.MSELoss()
        loss_data = loss_fn(pred[non_source_mask], y_norm[non_source_mask])
        

        
        # Note: the model output is du/dt, dv/dt, drho/dt, p
        # Now d/dt is just u(t+Δt)-u(t), need to divide by Δt, hard code Δt = 0.005
        #Later found dt=1, dx = 1 fits PDE better ### No, it just scales 
        dt = self.dt
        
        results_int_GT = self.phy_informer.forward(
            {
                "u": x[:, 0:1].reshape(batch_num, 1, xsize, ysize),
                "v": x[:, 1:2].reshape(batch_num, 1, xsize, ysize),
                "u_t": (y[:, 0:1]/dt).reshape(batch_num, 1, xsize, ysize),
                "v_t": (y[:, 1:2]/dt).reshape(batch_num, 1, xsize, ysize),
                "rho_t": (y[:, 2:3]/dt).reshape(batch_num, 1, xsize, ysize),
                "p_t": ((mgn_output[:, 3:4] - pres_hist)/dt).reshape(batch_num, 1, xsize, ysize),
                "rho": x[:, 2:3].reshape(batch_num, 1, xsize, ysize),
                "p": pres_hist.reshape(batch_num, 1, xsize, ysize)
            }
        )
        
        results_int = self.phy_informer.forward(
            {
                 "u": x[:, 0:1].reshape(batch_num, 1, xsize, ysize),
                "v": x[:, 1:2].reshape(batch_num, 1, xsize, ysize),
                "u_t": (mgn_output[:, 0:1]/dt).reshape(batch_num, 1, xsize, ysize),
                "v_t": (mgn_output[:, 1:2]/dt).reshape(batch_num, 1, xsize, ysize),
                "rho_t": (mgn_output[:, 2:3]/dt).reshape(batch_num, 1, xsize, ysize),
                "p_t": ((mgn_output[:, 3:4] - pres_hist)/dt).reshape(batch_num, 1, xsize, ysize),
                "rho": x[:, 2:3].reshape(batch_num, 1, xsize, ysize),
                "p": pres_hist.reshape(batch_num, 1, xsize, ysize)
            }
        )

        res_ideal, res_cont, res_mx, res_my=( 
                results_int["ideal_gas"] - results_int_GT["ideal_gas"],  #plaeceholder for ideal gas law equation
                results_int["continuity"] - results_int_GT["continuity"],
                results_int["momentum_x"] - results_int_GT["momentum_x"], 
                results_int["momentum_y"] - results_int_GT["momentum_y"], 
            )
        
        non_source_mask = non_source_mask.reshape(batch_num, 1, xsize, ysize)
        
        loss_ideal = torch.mean((res_ideal[non_source_mask])** 2 )
        loss_cont = torch.mean((res_cont[non_source_mask])** 2 )
        loss_mx = torch.mean((res_mx[non_source_mask])** 2 )
        loss_my = torch.mean((res_my[non_source_mask])** 2 )
        loss = {
                'ideal_gas':loss_ideal,
                'continuity': loss_cont,
                'momentum_x': loss_mx, 
                'momentum_y': loss_my,
                'data': loss_data
                }
        return loss
        
    def GT_res_backward(self, data_list, model, parallel, non_source_mask, train):
    
        if train:
            xsize = self.xsize
            ysize = self.ysize
        elif not train:
            xsize = self.full_xsize
            ysize = self.full_ysize
        # Note: pred must be denormalised before using
        pred = model(data_list)
       
        if model.normalize_output:
            mgn_output = model._output_normalizer.inverse(pred)
        else:
            mgn_output = pred


        batch_num = len(data_list.ptr)-1

        if parallel:
            #y is ground truth data
            y = cat([data.y for data in data_list]).to(pred.device)
            x = cat([data.x for data in data_list]).to(pred.device)

        else:
            y = data_list.y
            x = data_list.x
            pres_hist = data_list.pressure
        
        assert hasattr(model, "_output_normalizer")
        y_norm = model._output_normalizer(y, accumulate=model.training)
        #y_norm = y_norm[:, :4] # used for sedov, riemann
        
        
        loss_fn = nn.MSELoss()
        loss_data = loss_fn(pred[non_source_mask], y_norm[non_source_mask])
        
        #print('writing data into hdf5')
        
        #if not dist.is_initialized() or dist.get_rank() == 0:
        #    hdf5_dict = {
        #        'x': x, 
        #        'y': y, 
        #        'pres_hist': pres_hist, 
        #        'mgn_output': mgn_output,
        #        'non_source_mask' : non_source_mask
        #    }
        #    save_path = '/work4/clf/scarf1271/GNN/data_sets/Sedov/500_muscl_4pde_v2/loss_utils_sample_data.pt'
        #    torch.save(hdf5_dict, save_path)
        #print('writing done')
        # Note: the model output is du/dt, dv/dt, drho/dt, p
        # Now d/dt is just u(t+Δt)-u(t), need to divide by Δt, hard code Δt = 0.005
        #Later found dt=1, dx = 1 fits PDE better ### No, it just scales 
        dt = self.dt
        
        # Using backward Euler method --- more stable than forward
        u_GT = x[:, 0:1] + y[:, 0:1]
        v_GT = x[:, 1:2] + y[:, 1:2]
        rho_GT = x[:, 2:3] + y[:, 2:3]
        p_GT = y[:, 3:4]
        rho_u_t_GT = rho_GT * u_GT - (x[:, 2:3]) * (x[:, 0:1])
        rho_v_t_GT = rho_GT * v_GT - (x[:, 2:3]) * (x[:, 1:2])
        
        results_int_GT = self.phy_informer.forward(
            {
                "u": u_GT.reshape(batch_num, 1, xsize, ysize),
                "v": v_GT.reshape(batch_num, 1, xsize, ysize),
                "rho_u_t": (rho_u_t_GT/dt).reshape(batch_num, 1, xsize, ysize),
                "rho_v_t": (rho_v_t_GT/dt).reshape(batch_num, 1, xsize, ysize),
                "rho_t": (y[:, 2:3]/dt).reshape(batch_num, 1, xsize, ysize),
                "p_t": ((p_GT - pres_hist)/dt).reshape(batch_num, 1, xsize, ysize),
                "rho": rho_GT.reshape(batch_num, 1, xsize, ysize),
                "p": p_GT.reshape(batch_num, 1, xsize, ysize)
            }
        )
        
        u = x[:, 0:1] + mgn_output[:, 0:1]
        v = x[:, 1:2] + mgn_output[:, 1:2]
        rho = x[:, 2:3] + mgn_output[:, 2:3]
        p = mgn_output[:, 3:4]
        rho_u_t = (x[:, 2:3] + mgn_output[:, 2:3]) * (x[:, 0:1] + mgn_output[:, 0:1]) - (x[:, 2:3] * x[:, 0:1])
        rho_v_t = (x[:, 2:3] + mgn_output[:, 2:3]) * (x[:, 1:2] + mgn_output[:, 1:2]) - (x[:, 2:3] * x[:, 1:2])
        
        results_int = self.phy_informer.forward(
            {
                 "u": u.reshape(batch_num, 1, xsize, ysize),
                "v": v.reshape(batch_num, 1, xsize, ysize),
                "rho_u_t": (rho_u_t/dt).reshape(batch_num, 1, xsize, ysize),
                "rho_v_t": (rho_v_t/dt).reshape(batch_num, 1, xsize, ysize),
                "rho_t": (mgn_output[:, 2:3]/dt).reshape(batch_num, 1, xsize, ysize),
                "p_t": ((mgn_output[:, 3:4] - pres_hist)/dt).reshape(batch_num, 1, xsize, ysize),
                "rho": rho.reshape(batch_num, 1, xsize, ysize),
                "p": p.reshape(batch_num, 1, xsize, ysize)
            }
        )

        res_ideal, res_cont, res_mx, res_my=( 
                results_int["ideal_gas"] - results_int_GT["ideal_gas"],  #plaeceholder for ideal gas law equation
                results_int["continuity"] - results_int_GT["continuity"],
                results_int["momentum_x"] - results_int_GT["momentum_x"], 
                results_int["momentum_y"] - results_int_GT["momentum_y"], 
            )
        
        non_source_mask = non_source_mask.reshape(batch_num, 1, xsize, ysize)
        
        loss_ideal = torch.mean((res_ideal[non_source_mask])** 2 )
        loss_cont = torch.mean((res_cont[non_source_mask])** 2 )
        loss_mx = torch.mean((res_mx[non_source_mask])** 2 )
        loss_my = torch.mean((res_my[non_source_mask])** 2 )
        #print('loss_cont', loss_cont)
        #print('loss-mx', loss_mx)
        loss = {
                'ideal_gas':loss_ideal,
                'continuity': loss_cont,
                'momentum_x': loss_mx, 
                'momentum_y': loss_my,
                'data': loss_data
                }
        return loss
        
    
    
    def GT_res_backward_rel(self, data_list, model, parallel, non_source_mask, train):
    
        if train:
            xsize = self.xsize
            ysize = self.ysize
        elif not train:
            xsize = self.full_xsize
            ysize = self.full_ysize
        # Note: pred must be denormalised before using
        pred = model(data_list)
       
        if model.normalize_output:
            mgn_output = model._output_normalizer.inverse(pred)
        else:
            mgn_output = pred


        batch_num = len(data_list.ptr)-1

        if parallel:
            #y is ground truth data
            y = cat([data.y for data in data_list]).to(pred.device)
            x = cat([data.x for data in data_list]).to(pred.device)

        else:
            y = data_list.y
            x = data_list.x
            pres_hist = data_list.pressure
        
        assert hasattr(model, "_output_normalizer")
        y_norm = model._output_normalizer(y, accumulate=model.training)
        #y_norm = y_norm[:, :4] # used for sedov, riemann
        
        
        loss_fn = nn.MSELoss()
        loss_data = loss_fn(pred[non_source_mask], y_norm[non_source_mask])
        
        
        dt = self.dt
        
        # Using backward Euler method --- more stable than forward
        eps = 0.002
        
        u_t1_GT = x[:, 0:1] + y[:, 0:1]
        v_t1_GT = x[:, 1:2] + y[:, 1:2]
        rho_t1_GT = x[:, 2:3] + y[:, 2:3]
        p_t1_GT = y[:, 3:4]
        #gamma_t1_GT = 1#/(1-(u_t1_GT**2 + v_t1_GT**2))**0.5
        v2_t1_GT = u_t1_GT**2 + v_t1_GT**2
        gamma_t1_GT = 1/torch.sqrt(torch.clamp(1.0 - v2_t1_GT, min=eps))
        h_t1_GT = 1 + ( p_t1_GT * 1.4/0.4 )/rho_t1_GT
        
        
        u_t0_GT = x[:, 0:1] 
        v_t0_GT = x[:, 1:2] 
        rho_t0_GT = x[:, 2:3] 
        p_t0_GT = pres_hist
        
        #gamma_t0_GT = 1#/(1-(u_t0_GT**2 + v_t0_GT**2))**0.5
        v2_t0_GT = u_t0_GT**2 + v_t0_GT**2
        gamma_t0_GT = 1/torch.sqrt(torch.clamp(1.0 - v2_t0_GT, min=eps))
        h_t0_GT = 1 + ( p_t0_GT * 1.4/0.4 )/rho_t0_GT
        
        
        rho_rel_dt_GT = (rho_t1_GT*gamma_t1_GT - rho_t0_GT*gamma_t0_GT)/dt
        
        p_dt_GT = (p_t1_GT - p_t0_GT)/dt
        mom_u_t_GT = (rho_t1_GT*h_t1_GT*gamma_t1_GT**2*u_t1_GT - rho_t0_GT*h_t0_GT*gamma_t0_GT**2*u_t0_GT)/dt
        mom_v_t_GT = (rho_t1_GT*h_t1_GT*gamma_t1_GT**2*v_t1_GT - rho_t0_GT*h_t0_GT*gamma_t0_GT**2*v_t0_GT)/dt
        
        
        results_int_GT = self.phy_informer.forward(
            {
                "u": u_t1_GT.reshape(batch_num, 1, xsize, ysize),
                "v": v_t1_GT.reshape(batch_num, 1, xsize, ysize),
                "rho_t": rho_rel_dt_GT.reshape(batch_num, 1, xsize, ysize),
                "p_t": p_dt_GT.reshape(batch_num, 1, xsize, ysize),
                "rho": rho_t1_GT.reshape(batch_num, 1, xsize, ysize), 
                "p": p_t1_GT.reshape(batch_num, 1, xsize, ysize), 
                "mom_u_t":mom_u_t_GT.reshape(batch_num, 1, xsize, ysize), 
                "mom_v_t":mom_v_t_GT.reshape(batch_num, 1, xsize, ysize), 
            }
        )
        
        #u_t1 = x[:, 0:1] + mgn_output[:, 0:1]
        #v_t1 = x[:, 1:2] + mgn_output[:, 1:2]
        u_t1 = x[:, 0:1] + y[:, 0:1]
        v_t1 = x[:, 1:2] + y[:, 1:2]
        rho_t1 = x[:, 2:3] + mgn_output[:, 2:3]
        p_t1 = mgn_output[:, 3:4]
        #gamma_t1 = 1/(1-(u_t1**2 + v_t1**2))**0.5
        
        v2_t1 = u_t1**2 + v_t1**2
        gamma_t1 = 1/torch.sqrt(torch.clamp(1.0 - v2_t1, min=eps))
        h_t1 = 1 + ( p_t1 * 1.4/0.4 )/rho_t1
        
        # ---- NaN check for gamma_t1 ----
        #gamma_nan_mask = torch.isnan(gamma_t1)
        #gamma_nan_count = gamma_nan_mask.sum().item()
 
        #print("NaNs in gamma_t1:", gamma_nan_count)
        
        
        u_t0 = x[:, 0:1] 
        v_t0 = x[:, 1:2] 
        rho_t0 = x[:, 2:3] 
        p_t0 = pres_hist
        #gamma_t0 = 1/(1-(u_t0**2 + v_t0**2))**0.5
        v2_t0 = u_t0**2 + v_t0**2
        gamma_t0 = 1/torch.sqrt(torch.clamp(1.0 - v2_t0, min=eps))
        h_t0 = 1 + ( p_t0 * 1.4/0.4 )/rho_t0
        
        
        rho_rel_dt = (rho_t1*gamma_t1 - rho_t0*gamma_t0)/dt
        
        p_dt = (p_t1 - p_t0)/dt
        mom_u_t = (rho_t1*h_t1*gamma_t1**2*u_t1 - rho_t0*h_t0*gamma_t0**2*u_t0)/dt
        mom_v_t = (rho_t1*h_t1*gamma_t1**2*v_t1 - rho_t0*h_t0*gamma_t0**2*v_t0)/dt
        
        rho_rel_nan_count = torch.isnan(rho_rel_dt).sum().item()
        mom_u_t_nan_count = torch.isnan(mom_u_t).sum().item()
        mom_v_t_nan_count = torch.isnan(mom_v_t).sum().item()
        
 
        print("NaNs in rho_rel, mum_u, mom_v:", rho_rel_nan_count,mom_u_t_nan_count, mom_v_t_nan_count )
        
        
        results_int = self.phy_informer.forward(
            {
                "u": u_t1.reshape(batch_num, 1, xsize, ysize),
                "v": v_t1.reshape(batch_num, 1, xsize, ysize),
                "rho_t": rho_rel_dt.reshape(batch_num, 1, xsize, ysize),
                "p_t": p_dt.reshape(batch_num, 1, xsize, ysize),
                "rho": rho_t1.reshape(batch_num, 1, xsize, ysize), 
                "p": p_t1.reshape(batch_num, 1, xsize, ysize), 
                "mom_u_t":mom_u_t.reshape(batch_num, 1, xsize, ysize), 
                "mom_v_t":mom_v_t.reshape(batch_num, 1, xsize, ysize), 
            }
        )
        



        res_ideal, res_cont, res_mx, res_my=( 
                results_int["ideal_gas"] - results_int_GT["ideal_gas"],  #plaeceholder for ideal gas law equation
                results_int["continuity"] - results_int_GT["continuity"],
                results_int["momentum_x"] - results_int_GT["momentum_x"], 
                results_int["momentum_y"] - results_int_GT["momentum_y"], 
            )
            
        print('res-GT_cont Nan count', torch.isnan(results_int_GT["continuity"]).sum().item()) #results_int_GT["continuity"], results_int["continuity"])
        print('res_cont Nan count', torch.isnan(results_int["continuity"]).sum().item())
        print('res_GT_mx Nan count', torch.isnan(results_int_GT["momentum_x"]).sum().item()) #results_int_GT["momentum_x"], results_int["continuity"])
        print('res_mx Nan count', torch.isnan(results_int["momentum_x"]).sum().item())
        non_source_mask = non_source_mask.reshape(batch_num, 1, xsize, ysize)
        
        loss_ideal = torch.nanmean((res_ideal[non_source_mask])** 2 )
        loss_cont = torch.nanmean((res_cont[non_source_mask])** 2 )
        loss_mx = torch.nanmean((res_mx[non_source_mask])** 2 )
        loss_my = torch.nanmean((res_my[non_source_mask])** 2 )
        
        print('loss_cont', loss_cont)
        
        print('loss_mx', loss_mx)
        
        loss = {
                'ideal_gas':loss_ideal,
                'continuity': loss_cont,
                'momentum_x': loss_mx, 
                'momentum_y': loss_my,
                'data': loss_data
                }
        return loss
        
        
    
    def GT_res_backward_weighted_data(self, data_list, model, parallel, non_source_mask, train):
    
        if train:
            xsize = self.xsize
            ysize = self.ysize
        elif not train:
            xsize = self.full_xsize
            ysize = self.full_ysize
        # Note: pred must be denormalised before using
        pred = model(data_list)
       
        if model.normalize_output:
            mgn_output = model._output_normalizer.inverse(pred)
        else:
            mgn_output = pred


        batch_num = len(data_list.ptr)-1

        if parallel:
            #y is ground truth data
            y = cat([data.y for data in data_list]).to(pred.device)
            x = cat([data.x for data in data_list]).to(pred.device)

        else:
            y = data_list.y
            x = data_list.x
            pres_hist = data_list.pressure
        
        assert hasattr(model, "_output_normalizer")
        y_norm = model._output_normalizer(y, accumulate=model.training)
        #y_norm = y_norm[:, :4] # used for sedov, riemann
        
        
        loss_fn = nn.MSELoss()
        velx_data = loss_fn(pred[non_source_mask][:, 0], y_norm[non_source_mask][:, 0])
        vely_data = loss_fn(pred[non_source_mask][:, 1], y_norm[non_source_mask][:, 1])
        dens_data = loss_fn(pred[non_source_mask][:, 2], y_norm[non_source_mask][:, 2])
        pres_data = loss_fn(pred[non_source_mask][:, 3], y_norm[non_source_mask][:, 3])
        
        #loss_data = 1 * velx_data + 1 * vely_data + 1 * dens_data + 3 * pres_data    #used 1 in _wd, 3 for pres in wd_10p, 10 for uv in wd_10uv
        
        loss_data = 1 * velx_data + 1 * vely_data + 1 * dens_data + 3 * pres_data
        #print('writing data into hdf5')
        
        #if not dist.is_initialized() or dist.get_rank() == 0:
        #    hdf5_dict = {
        #        'x': x, 
        #        'y': y, 
        #        'pres_hist': pres_hist, 
        #        'mgn_output': mgn_output,
        #        'non_source_mask' : non_source_mask
        #    }
        #    save_path = '/work4/clf/scarf1271/GNN/data_sets/Sedov/500_muscl_4pde_v2/loss_utils_sample_data.pt'
        #    torch.save(hdf5_dict, save_path)
        #print('writing done')
        # Note: the model output is du/dt, dv/dt, drho/dt, p
        # Now d/dt is just u(t+Δt)-u(t), need to divide by Δt, hard code Δt = 0.005
        #Later found dt=1, dx = 1 fits PDE better ### No, it just scales 
        dt = self.dt
        
        # Using backward Euler method --- more stable than forward
        u_GT = x[:, 0:1] + y[:, 0:1]
        v_GT = x[:, 1:2] + y[:, 1:2]
        rho_GT = x[:, 2:3] + y[:, 2:3]
        p_GT = y[:, 3:4]
        rho_u_t_GT = rho_GT * u_GT - (x[:, 2:3]) * (x[:, 0:1])
        rho_v_t_GT = rho_GT * v_GT - (x[:, 2:3]) * (x[:, 1:2])
        
        results_int_GT = self.phy_informer.forward(
            {
                "u": u_GT.reshape(batch_num, 1, xsize, ysize),
                "v": v_GT.reshape(batch_num, 1, xsize, ysize),
                "rho_u_t": (rho_u_t_GT/dt).reshape(batch_num, 1, xsize, ysize),
                "rho_v_t": (rho_v_t_GT/dt).reshape(batch_num, 1, xsize, ysize),
                "rho_t": (y[:, 2:3]/dt).reshape(batch_num, 1, xsize, ysize),
                "p_t": ((p_GT - pres_hist)/dt).reshape(batch_num, 1, xsize, ysize),
                "rho": rho_GT.reshape(batch_num, 1, xsize, ysize),
                "p": p_GT.reshape(batch_num, 1, xsize, ysize)
            }
        )
        
        u = x[:, 0:1] + mgn_output[:, 0:1]
        v = x[:, 1:2] + mgn_output[:, 1:2]
        rho = x[:, 2:3] + mgn_output[:, 2:3]
        p = mgn_output[:, 3:4]
        rho_u_t = (x[:, 2:3] + mgn_output[:, 2:3]) * (x[:, 0:1] + mgn_output[:, 0:1]) - (x[:, 2:3] * x[:, 0:1])
        rho_v_t = (x[:, 2:3] + mgn_output[:, 2:3]) * (x[:, 1:2] + mgn_output[:, 1:2]) - (x[:, 2:3] * x[:, 1:2])
        
        results_int = self.phy_informer.forward(
            {
                 "u": u.reshape(batch_num, 1, xsize, ysize),
                "v": v.reshape(batch_num, 1, xsize, ysize),
                "rho_u_t": (rho_u_t/dt).reshape(batch_num, 1, xsize, ysize),
                "rho_v_t": (rho_v_t/dt).reshape(batch_num, 1, xsize, ysize),
                "rho_t": (mgn_output[:, 2:3]/dt).reshape(batch_num, 1, xsize, ysize),
                "p_t": ((mgn_output[:, 3:4] - pres_hist)/dt).reshape(batch_num, 1, xsize, ysize),
                "rho": rho.reshape(batch_num, 1, xsize, ysize),
                "p": p.reshape(batch_num, 1, xsize, ysize)
            }
        )

        res_ideal, res_cont, res_mx, res_my=( 
                results_int["ideal_gas"] - results_int_GT["ideal_gas"],  #plaeceholder for ideal gas law equation
                results_int["continuity"] - results_int_GT["continuity"],
                results_int["momentum_x"] - results_int_GT["momentum_x"], 
                results_int["momentum_y"] - results_int_GT["momentum_y"], 
            )
        
        non_source_mask = non_source_mask.reshape(batch_num, 1, xsize, ysize)
        
        loss_ideal = torch.mean((res_ideal[non_source_mask])** 2 )
        loss_cont = torch.mean((res_cont[non_source_mask])** 2 )
        loss_mx = torch.mean((res_mx[non_source_mask])** 2 )
        loss_my = torch.mean((res_my[non_source_mask])** 2 )
        #print('loss_cont', loss_cont)
        #print('loss-mx', loss_mx)
        loss = {
                'ideal_gas':loss_ideal,
                'continuity': loss_cont,
                'momentum_x': loss_mx, 
                'momentum_y': loss_my,
                'data': loss_data
                }
        return loss
        
    
    def GT_res_backward_weighted_data_10p(self, data_list, model, parallel, non_source_mask, train):
    
        if train:
            xsize = self.xsize
            ysize = self.ysize
        elif not train:
            xsize = self.full_xsize
            ysize = self.full_ysize
        # Note: pred must be denormalised before using
        pred = model(data_list)
       
        if model.normalize_output:
            mgn_output = model._output_normalizer.inverse(pred)
        else:
            mgn_output = pred


        batch_num = len(data_list.ptr)-1

        if parallel:
            #y is ground truth data
            y = cat([data.y for data in data_list]).to(pred.device)
            x = cat([data.x for data in data_list]).to(pred.device)

        else:
            y = data_list.y
            x = data_list.x
            pres_hist = data_list.pressure
        
        assert hasattr(model, "_output_normalizer")
        y_norm = model._output_normalizer(y, accumulate=model.training)
        #y_norm = y_norm[:, :4] # used for sedov, riemann
        
        
        loss_fn = nn.MSELoss()
        velx_data = loss_fn(pred[non_source_mask][:, 0], y_norm[non_source_mask][:, 0])
        vely_data = loss_fn(pred[non_source_mask][:, 1], y_norm[non_source_mask][:, 1])
        dens_data = loss_fn(pred[non_source_mask][:, 2], y_norm[non_source_mask][:, 2])
        pres_data = loss_fn(pred[non_source_mask][:, 3], y_norm[non_source_mask][:, 3])
        
        #loss_data = 1 * velx_data + 1 * vely_data + 1 * dens_data + 3 * pres_data    #used 1 in _wd, 10 for pres in wd_10p, 3 for pres in wd_3p, 10 for uv in wd_10uv
        
        loss_data = 1 * velx_data + 1 * vely_data + 1 * dens_data + 10 * pres_data
        #print('writing data into hdf5')
        
        #if not dist.is_initialized() or dist.get_rank() == 0:
        #    hdf5_dict = {
        #        'x': x, 
        #        'y': y, 
        #        'pres_hist': pres_hist, 
        #        'mgn_output': mgn_output,
        #        'non_source_mask' : non_source_mask
        #    }
        #    save_path = '/work4/clf/scarf1271/GNN/data_sets/Sedov/500_muscl_4pde_v2/loss_utils_sample_data.pt'
        #    torch.save(hdf5_dict, save_path)
        #print('writing done')
        # Note: the model output is du/dt, dv/dt, drho/dt, p
        # Now d/dt is just u(t+Δt)-u(t), need to divide by Δt, hard code Δt = 0.005
        #Later found dt=1, dx = 1 fits PDE better ### No, it just scales 
        dt = self.dt
        
        # Using backward Euler method --- more stable than forward
        u_GT = x[:, 0:1] + y[:, 0:1]
        v_GT = x[:, 1:2] + y[:, 1:2]
        rho_GT = x[:, 2:3] + y[:, 2:3]
        p_GT = y[:, 3:4]
        rho_u_t_GT = rho_GT * u_GT - (x[:, 2:3]) * (x[:, 0:1])
        rho_v_t_GT = rho_GT * v_GT - (x[:, 2:3]) * (x[:, 1:2])
        
        results_int_GT = self.phy_informer.forward(
            {
                "u": u_GT.reshape(batch_num, 1, xsize, ysize),
                "v": v_GT.reshape(batch_num, 1, xsize, ysize),
                "rho_u_t": (rho_u_t_GT/dt).reshape(batch_num, 1, xsize, ysize),
                "rho_v_t": (rho_v_t_GT/dt).reshape(batch_num, 1, xsize, ysize),
                "rho_t": (y[:, 2:3]/dt).reshape(batch_num, 1, xsize, ysize),
                "p_t": ((p_GT - pres_hist)/dt).reshape(batch_num, 1, xsize, ysize),
                "rho": rho_GT.reshape(batch_num, 1, xsize, ysize),
                "p": p_GT.reshape(batch_num, 1, xsize, ysize)
            }
        )
        
        u = x[:, 0:1] + mgn_output[:, 0:1]
        v = x[:, 1:2] + mgn_output[:, 1:2]
        rho = x[:, 2:3] + mgn_output[:, 2:3]
        p = mgn_output[:, 3:4]
        rho_u_t = (x[:, 2:3] + mgn_output[:, 2:3]) * (x[:, 0:1] + mgn_output[:, 0:1]) - (x[:, 2:3] * x[:, 0:1])
        rho_v_t = (x[:, 2:3] + mgn_output[:, 2:3]) * (x[:, 1:2] + mgn_output[:, 1:2]) - (x[:, 2:3] * x[:, 1:2])
        
        results_int = self.phy_informer.forward(
            {
                 "u": u.reshape(batch_num, 1, xsize, ysize),
                "v": v.reshape(batch_num, 1, xsize, ysize),
                "rho_u_t": (rho_u_t/dt).reshape(batch_num, 1, xsize, ysize),
                "rho_v_t": (rho_v_t/dt).reshape(batch_num, 1, xsize, ysize),
                "rho_t": (mgn_output[:, 2:3]/dt).reshape(batch_num, 1, xsize, ysize),
                "p_t": ((mgn_output[:, 3:4] - pres_hist)/dt).reshape(batch_num, 1, xsize, ysize),
                "rho": rho.reshape(batch_num, 1, xsize, ysize),
                "p": p.reshape(batch_num, 1, xsize, ysize)
            }
        )

        res_ideal, res_cont, res_mx, res_my=( 
                results_int["ideal_gas"] - results_int_GT["ideal_gas"],  #plaeceholder for ideal gas law equation
                results_int["continuity"] - results_int_GT["continuity"],
                results_int["momentum_x"] - results_int_GT["momentum_x"], 
                results_int["momentum_y"] - results_int_GT["momentum_y"], 
            )
        
        non_source_mask = non_source_mask.reshape(batch_num, 1, xsize, ysize)
        
        loss_ideal = torch.mean((res_ideal[non_source_mask])** 2 )
        loss_cont = torch.mean((res_cont[non_source_mask])** 2 )
        loss_mx = torch.mean((res_mx[non_source_mask])** 2 )
        loss_my = torch.mean((res_my[non_source_mask])** 2 )
        #print('loss_cont', loss_cont)
        #print('loss-mx', loss_mx)
        loss = {
                'ideal_gas':loss_ideal,
                'continuity': loss_cont,
                'momentum_x': loss_mx, 
                'momentum_y': loss_my,
                'data': loss_data
                }
        return loss
        
        
    def data_single_parameter(self, data_list, model, parallel, non_source_mask, train):
    
        if train:
            xsize = self.xsize
            ysize = self.ysize
        elif not train:
            xsize = self.full_xsize
            ysize = self.full_ysize
        # Note: pred must be denormalised before using
        pred = model(data_list)
       
        if model.normalize_output:
            mgn_output = model._output_normalizer.inverse(pred)
        else:
            mgn_output = pred


        batch_num = len(data_list.ptr)-1

        if parallel:
            #y is ground truth data
            y = cat([data.y for data in data_list]).to(pred.device)
            x = cat([data.x for data in data_list]).to(pred.device)

        else:
            y = data_list.y
            x = data_list.x
            
        
        assert hasattr(model, "_output_normalizer")
        y_norm = model._output_normalizer(y, accumulate=model.training)
        #y_norm = y_norm[:, :4] # used for sedov, riemann
        
        
        loss_fn = nn.MSELoss()
        loss_data = loss_fn(pred[non_source_mask], y_norm[non_source_mask])
        
        
        
        
        
        loss_ideal = tensor(0.0)
        loss_cont = tensor(0.0)
        loss_mx = tensor(0.0)
        loss_my = tensor(0.0)
        
        loss = {
                'ideal_gas':loss_ideal,
                'continuity': loss_cont,
                'momentum_x': loss_mx, 
                'momentum_y': loss_my,
                'data': loss_data
                }
        return loss
        
        
        
 
 # PINNLoss with adaptive loss weight
class PINNLoss_alw(nn.Module):
    def __init__(self, initial_log_alpha=-4.6, initial_log_beta=-0.01):
        """
        Compute combined PINN loss with self-adaptive weighting using log(alpha) and log(beta).

        Parameters:
        - initial_log_alpha (float): Initial value for log(alpha).
        - initial_log_beta (float): Initial value for log(beta).

        * Default values:
          - log(0.01) ≈ -4.6 (small initial continuity weight)
          - log(0.99) ≈ -0.01 (large initial MSE weight)
        """
        super(PINNLoss_alw, self).__init__()

        # Learnable log_alpha and log_beta
        self.log_alpha = nn.Parameter(torch.tensor(initial_log_alpha, dtype=torch.float32))
        self.log_beta = nn.Parameter(torch.tensor(initial_log_beta, dtype=torch.float32))

        

    def forward(self, pred, ground_truth, coords, mesh):
        """
        Compute the combined loss.

        Parameters:
        - pred (torch.Tensor): Predicted values [N, D].
        - ground_truth (torch.Tensor): Ground truth values [N, D].
        - coords (torch.Tensor): Node coordinates [N, 2].
        - mesh: Mesh structure used for computing gradients.

        Returns:
        - loss (torch.Tensor): Adaptive loss (weighted sum of continuity and MSE loss).
        - mse_loss (torch.Tensor): Data loss.
        - continuity_loss (torch.Tensor): PDE residual loss.
        - alpha (float): Current value of alpha.
        - beta (float): Current value of beta.
        """
        
        self.gradient_operator = GradientOperator(mesh)

        # Compute continuity (PDE) residual
        grad_uv = self.gradient_operator.compute_gradient(pred, coords)
        du_dx = grad_uv[:, 0, 0]
        dv_dy = grad_uv[:, 1, 1]
        continuity_residual = du_dx + dv_dy
        continuity_loss = torch.mean(continuity_residual ** 2)

        # Compute MSE loss
        mse_loss = torch.mean((pred - ground_truth) ** 2)

        # Convert log_alpha and log_beta to positive values
        alpha = torch.exp(self.log_alpha)  # Ensures alpha > 0
        beta = torch.exp(self.log_beta)  # Ensures beta > 0

        # Compute the final adaptive loss
        loss = (1 / (2 * alpha**2)) * continuity_loss + (1 / (2 * beta**2)) * mse_loss + self.log_alpha + self.log_beta

        return loss       
        
        
        
        
