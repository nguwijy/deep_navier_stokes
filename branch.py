import os
import time
import math
import torch
from torch.distributions.exponential import Exponential
import matplotlib.pyplot as plt
import numpy as np
import logging
from fdb import fdb_nd

torch.manual_seed(0)  # set seed for reproducibility


class Net(torch.nn.Module):
    """
    deep branching approach to solve PDE system with utility functions
    """

    def __init__(
        self,
        f_fun,
        deriv_map,
        zeta_map,
        deriv_condition_deriv_map,
        deriv_condition_zeta_map,
        phi_fun=(lambda x: x),
        exact_p_fun=None,
        exact_u_fun=None,
        x_lo=-10.0,
        x_hi=10.0,
        t_lo=0.0,
        t_hi=0.0,
        T=1.0,
        beta=0.5,
        branch_exponential_lambda=None,
        neurons=20,
        layers=5,
        branch_lr=1e-2,
        lr_milestones=[3000 // 2],
        lr_gamma=0.1,
        weight_decay=0,
        branch_nb_path_per_state=1000,
        branch_nb_states=10,
        branch_nb_states_per_batch=10,
        epochs=3000,
        batch_normalization=True,
        debug=False,
        antithetic=True,
        poisson_loss_coeff=0.,
        deriv_condition_coeff=1.,
        overtrain_rate=0.1,
        device="cpu",
        branch_activation="tanh",
        verbose=False,
        fix_all_dim_except_first=False,
        branch_patches=1,
        outlier_percentile=1,
        outlier_multiplier=10,
        plot_y_lim=None,
        quantization=False,
        **kwargs,
    ):
        super(Net, self).__init__()
        self.f_fun = f_fun
        self.phi_fun = phi_fun
        self.exact_p_fun = exact_p_fun
        self.exact_u_fun = exact_u_fun
        self.deriv_map = deriv_map
        self.zeta_map = zeta_map
        self.deriv_condition_deriv_map = deriv_condition_deriv_map
        self.deriv_condition_zeta_map = deriv_condition_zeta_map
        self.n, self.dim = deriv_map.shape
        self.nprime = sum(zeta_map == -1)
        self.patches = branch_patches

        self.quantization = quantization
        if quantization:
            from scipy import spatial
            pts = 100
            tmp = np.loadtxt(f"quantization/{pts}_{self.dim}_nopti")
            self.quant_prob = torch.tensor(
                    tmp[:pts, 0],
                    device=device,
                    dtype=torch.get_default_dtype()
            )
            self.quant_grids_tree = spatial.KDTree(tmp[:pts, 1:(self.dim+1)])
            self.quant_grids_tensor = torch.tensor(
                    self.quant_grids_tree.data,
                    device=device,
                    dtype=torch.get_default_dtype()
            )

        # store the (faa di bruno) fdb results for quicker lookup
        start = time.time()
        self.fdb_lookup = {
            tuple(deriv): fdb_nd(self.n, tuple(deriv))
            for deriv in deriv_map[self.nprime:]
        }
        self.fdb_runtime = time.time() - start
        self.mechanism_tot_len = (
            self.dim * self.n**2
            + self.nprime
            + sum(
                [
                    len(self.fdb_lookup[tuple(deriv)])
                    for deriv in deriv_map[self.nprime:]
                ]
            )
        )

        self.u_layer = torch.nn.ModuleList(
            [
                torch.nn.ModuleList(
                    [torch.nn.Linear(self.dim + 1, neurons, device=device)]
                    + [
                        torch.nn.Linear(neurons, neurons, device=device)
                        for _ in range(layers)
                    ]
                    + [torch.nn.Linear(neurons, self.dim, device=device)]
                )
                for _ in range(branch_patches)
            ]
        )
        self.u_bn_layer = torch.nn.ModuleList(
            [
                torch.nn.ModuleList(
                    [torch.nn.BatchNorm1d(self.dim + 1, device=device)]
                    + [
                        torch.nn.BatchNorm1d(neurons, device=device)
                        for _ in range(layers + 1)
                    ]
                )
                for _ in range(branch_patches)
            ]
        )
        self.p_layer = torch.nn.ModuleList(
            [
                torch.nn.ModuleList(
                    [torch.nn.Linear(self.dim, neurons, device=device)]
                    + [
                        torch.nn.Linear(neurons, neurons, device=device)
                        for _ in range(layers)
                    ]
                    + [torch.nn.Linear(neurons, 1, device=device)]
                )
                for _ in range(branch_patches)
            ]
        )
        self.p_bn_layer = torch.nn.ModuleList(
            [
                torch.nn.ModuleList(
                    [torch.nn.BatchNorm1d(self.dim, device=device)]
                    + [
                        torch.nn.BatchNorm1d(neurons, device=device)
                        for _ in range(layers + 1)
                    ]
                )
                for _ in range(branch_patches)
            ]
        )
        self.loaded_dict = False
        self.lr = branch_lr
        self.lr_milestones = lr_milestones
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.loss = torch.nn.MSELoss()
        self.activation = {
            "tanh": torch.nn.Tanh(),
            "relu": torch.nn.ReLU(),
            "softplus": torch.nn.ReLU(),
        }[branch_activation]
        self.batch_normalization = batch_normalization
        self.debug = debug
        self.nb_states = branch_nb_states
        self.nb_states_per_batch = branch_nb_states_per_batch
        self.nb_path_per_state = branch_nb_path_per_state
        self.x_lo = x_lo
        self.x_hi = x_hi
        # slight overtrain the domain of x for higher precision near boundary
        self.adjusted_x_boundaries = (
            x_lo - overtrain_rate * (x_hi - x_lo),
            x_hi + overtrain_rate * (x_hi - x_lo),
        )
        self.t_lo = t_lo
        self.t_hi = t_lo if fix_all_dim_except_first else t_hi
        self.T = T
        self.tau_lo, self.tau_hi = 1e-5, 6  # for negative coordinate
        self.beta = beta
        self.delta_t = (T - t_lo) / branch_patches
        self.outlier_percentile = outlier_percentile
        self.outlier_multiplier = outlier_multiplier

        self.exponential_lambda = (
            branch_exponential_lambda
            if branch_exponential_lambda is not None
            else -math.log(0.95) / T
        )
        self.epochs = epochs
        self.antithetic = antithetic
        self.poisson_loss_coeff = poisson_loss_coeff
        self.deriv_condition_coeff = deriv_condition_coeff
        self.device = device
        self.verbose = verbose
        self.fix_all_dim_except_first = fix_all_dim_except_first
        self.t_boundaries = torch.tensor(
            ([t_lo + i * self.delta_t for i in range(branch_patches)] + [T])[::-1],
            device=device,
        )
        self.adjusted_t_boundaries = [
            (lo, hi) for hi, lo in zip(self.t_boundaries[:-1], self.t_boundaries[1:])
        ]
        timestr = time.strftime("%Y%m%d-%H%M%S")  # current time stamp
        self.working_dir = f"logs/{timestr}"
        self.log_config()
        self.plot_y_lim = plot_y_lim

    def calculate_p_from_u_quant(self, x):
        x = x.detach().clone()
        dt = (self.tau_hi - self.tau_lo)/100
        tnow = self.tau_lo
        ans, adj = 0, 0
        while tnow + 1e-10 < self.tau_hi:
            exact_y = self.quant_grids_tensor.data.unsqueeze(1).repeat(1, x.shape[0], 1)
            # add y to x
            exact_xy = (x + math.sqrt(tnow) * exact_y).reshape(-1, self.dim).T.requires_grad_(True)
            # reshape y
            exact_y = math.sqrt(tnow) * exact_y.reshape(-1, self.dim)
            # multiplier
            exact_multiplier = dt * (exact_y ** 2).sum(dim=-1) / (2 * tnow)
            if self.dim > 2:
                exact_multiplier /= (self.dim - 2)
            elif self.dim == 2:
                exact_multiplier *= -torch.log((exact_y ** 2).sum(dim=-1).sqrt())
            order = np.array([0] * self.dim)
            for i in range(self.dim):
                for j in range(self.dim):
                    order[i] += 1
                    tmp3 = self.nth_derivatives(
                        order, self.phi_fun(exact_xy, j), exact_xy
                    )
                    order[i] -= 1
                    order[j] += 1
                    tmp3 *= self.nth_derivatives(
                        order, self.phi_fun(exact_xy, i), exact_xy
                    )
                    order[j] -= 1
                    ans += exact_multiplier * tmp3
            tnow += dt
        ans = (ans.reshape(-1, x.shape[0]).T * self.quant_prob).sum(dim=-1)
        return ans.detach()

    def calculate_p_from_u(self, x):
        x = x.detach().clone().requires_grad_(True)
        nb_mc = self.nb_path_per_state
        x = x.repeat(nb_mc, 1, 1)
        unif = (
            torch.rand(nb_mc * x.shape[1], device=self.device)
                 .reshape(nb_mc, -1, 1)
        )
        tau = self.tau_lo + (self.tau_hi - self.tau_lo) * unif
        y = self.gen_bm(tau.transpose(0, -1), x.shape[1], var=1).transpose(0, -1)
        x = x + y
        x = x.reshape(-1, self.dim).T
        order = np.array([0] * self.dim)
        ans = 0
        for i in range(self.dim):
            for j in range(self.dim):
                order[i] += 1
                tmp = self.nth_derivatives(
                    order, self.phi_fun(x, j), x
                )
                order[i] -= 1
                order[j] += 1
                tmp *= self.nth_derivatives(
                    order, self.phi_fun(x, i), x
                )
                order[j] -= 1
                ans += tmp
        ans = ans.reshape(nb_mc, -1)
        ans *= (y**2).sum(dim=-1)
        if self.dim > 2:
            ans /= (self.dim - 2)
        elif self.dim == 2:
            ans *= -torch.log((y**2).sum(dim=-1).sqrt())
        ans *= ((self.tau_hi - self.tau_lo) / (2 * tau[:, :, 0]))
        return ans.mean(dim=0).detach()

    def forward(self, x, patch=None, p_or_u="u"):
        """
        self(x) evaluates the neural network approximation NN(x)
        """
        layer = self.u_layer if p_or_u == "u" else self.p_layer
        bn_layer = self.u_bn_layer if p_or_u == "u" else self.p_bn_layer

        if self.debug:
            # return the exact function for debug purposes
            if p_or_u == "p":
                return self.exact_p_fun(torch.cat([self.T * torch.ones_like(x[:, :1]), x], dim=-1))

        if patch is not None:
            y = x
            if self.batch_normalization:
                y = bn_layer[patch][0](y)
            for idx, (f, bn) in enumerate(zip(layer[patch][:-1], bn_layer[patch][1:])):
                tmp = f(y)
                tmp = self.activation(tmp)
                if self.batch_normalization:
                    tmp = bn(tmp)
                if idx == 0:
                    y = tmp
                else:
                    # resnet
                    y = tmp + y

            y = layer[patch][-1](y)
        else:
            yy = []
            for p in range(self.patches):
                y = x
                if self.batch_normalization:
                    y = bn_layer[p][0](y)
                for idx, (f, bn) in enumerate(zip(layer[p][:-1], bn_layer[p][1:])):
                    tmp = f(y)
                    tmp = self.activation(tmp)
                    if self.batch_normalization:
                        tmp = bn(tmp)
                    if idx == 0:
                        y = tmp
                    else:
                        # resnet
                        y = tmp + y
                yy.append(layer[p][-1](y))
            idx = self.bisect_left(x[:, 0])
            y = torch.gather(torch.stack(yy, dim=-1), -1, idx.reshape(-1, 1)).squeeze(
                -1
            )
        return y

    def log_config(self):
        """
        set up configuration for log files and mkdir
        """
        os.mkdir(self.working_dir)
        os.mkdir(f"{self.working_dir}/model")
        os.mkdir(f"{self.working_dir}/plot")
        os.mkdir(f"{self.working_dir}/plot/p")
        for i in range(self.dim):
            os.mkdir(f"{self.working_dir}/plot/u{i}")
        formatter = "%(asctime)s | %(name)s |  %(levelname)s: %(message)s"
        logging.basicConfig(
            filename=f"{self.working_dir}/testrun.log",
            filemode="w",
            level=logging.DEBUG,
            format=formatter,
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger().addHandler(console)
        logging.debug(f"Current configuration: {self.__dict__}")

    def bisect_left(self, val):
        """
        find the index of val based on the discretization of self.t_boundaries
        it is only used when branch_patches > 1
        """
        idx = (
            torch.max(self.t_boundaries <= (val + 1e-8).reshape(-1, 1), dim=1)[
                1
            ].reshape(val.shape)
            - 1
        )
        # t_boundaries[0], use the first network
        idx = idx.where(~(val == self.t_boundaries[0]), torch.zeros_like(idx))
        # t_boundaries[-1], use the last network
        idx = idx.where(
            ~(val == self.t_boundaries[-1]),
            (self.t_boundaries.shape[0] - 2) * torch.ones_like(idx),
        )
        return idx

    @staticmethod
    def pretty_print(tensor):
        mess = ""
        for i in tensor[:-1]:
            mess += f"& {i.item():.2E} "
        mess += "& --- \\\\"
        logging.info(mess)

    def error_calculation(self, nb_pts_time=11, nb_pts_spatial=2*126+1, error_multiplier=1):
        self.eval()
        x = np.linspace(self.x_lo, self.x_hi, nb_pts_spatial)
        t = np.linspace(self.t_lo, self.t_hi, nb_pts_time)
        arr = np.array(np.meshgrid(*([x]*self.dim + [t]))).T.reshape(-1, self.dim + 1)
        arr[:, [-1, 0]] = arr[:, [0, -1]]
        arr = torch.tensor(arr, device=device, dtype=torch.get_default_dtype())
        error = []
        nn = []
        cur, batch_size, last = 0, 100000, arr.shape[0]
        while cur < last:
            nn.append(self(arr[cur:min(cur+batch_size, last)], patch=0).detach())
            cur += batch_size
        nn = torch.cat(nn, dim=0)

        # Lejay
        logging.info("The error as in Lejay is calculated as follows.")
        overall_error = 0
        for i in range(self.dim):
            error.append(error_multiplier * (nn[:, i] - self.exact_u_fun(arr, i)).reshape(nb_pts_time, -1) ** 2)
            overall_error += (error[-1])
        error.append(overall_error)
        for i in range(self.dim):
            logging.info(f"$\\hat{{e}}_{i}(t_k)$")
            self.pretty_print(error[i].max(dim=1)[0])
        logging.info("$\\hat{e}(t_k)$")
        self.pretty_print(error[-1].max(dim=1)[0])
        logging.info("\\hline")

        # erru
        logging.info("\nThe relative L2 error of u (erru) is calculated as follows.")
        denominator, numerator = 0, 0
        for i in range(self.dim):
            denominator += self.exact_u_fun(arr, i).reshape(nb_pts_time, -1) ** 2
            numerator += (nn[:, i] - self.exact_u_fun(arr, i)).reshape(nb_pts_time, -1) ** 2
        logging.info("erru($t_k$)")
        self.pretty_print((numerator.mean(dim=-1)/denominator.mean(dim=-1)).sqrt())

        del nn
        torch.cuda.empty_cache()
        grad = []
        cur, batch_size, last = 0, 100000, arr.shape[0]
        while cur < last:
            xx = arr[cur:min(cur+batch_size, last)].detach().clone().requires_grad_(True)
            tmp = []
            for i in range(self.dim):
                tmp.append(
                    torch.autograd.grad(
                        self(xx, patch=0)[:, i].sum(),
                        xx,
                    )[0][:, 1:].detach()
                )
            grad.append(torch.stack(tmp, dim=-1))
            cur += batch_size
        grad = torch.cat(grad, dim=0)

        # errgu
        logging.info("\nThe relative L2 error of gradient of u (errgu) is calculated as follows.")
        denominator, numerator = 0, 0
        xx = arr.detach().clone().requires_grad_(True)
        for i in range(self.dim):
            exact = torch.autograd.grad(
                    self.exact_u_fun(xx, i).sum(),
                    xx,
            )[0][:, 1:]
            denominator += exact.reshape(nb_pts_time, -1, self.dim) ** 2
            numerator += (exact - grad[:, :, i]).reshape(nb_pts_time, -1, self.dim) ** 2
        logging.info("errgu($t_k$)")
        self.pretty_print((numerator.mean(dim=(1, 2))/denominator.mean(dim=(1, 2))).sqrt())

        # errdivu
        logging.info("\nThe absolute divergence of u (errdivu) is calculated as follows.")
        numerator = 0
        for i in range(self.dim):
            numerator += (grad[:, i, i]).reshape(nb_pts_time, -1)
        numerator = numerator**2
        logging.info("errdivu($t_k$)")
        self.pretty_print(
                ((self.x_hi - self.x_lo)**self.dim * numerator.mean(dim=-1)).sqrt()
        )

        del grad, xx
        torch.cuda.empty_cache()
        arr = arr.reshape(nb_pts_time, -1, self.dim + 1)[-1].detach()
        nn = []
        cur, batch_size, last = 0, 100000, arr.shape[0]
        while cur < last:
            nn.append(
                self(
                    arr[cur:min(cur+batch_size, last), 1:],
                    patch=0,
                    p_or_u="p",
                ).squeeze().detach()
            )
            cur += batch_size
        nn = torch.cat(nn, dim=0)

        # errp
        logging.info("\nThe relative L2 error of p (errp) is calculated as follows.")
        denominator = (self.exact_p_fun(arr) - self.exact_p_fun(arr).mean()) ** 2
        numerator = (
                nn - nn.mean() - self.exact_p_fun(arr) + self.exact_p_fun(arr).mean()
        ) ** 2
        logging.info("errp($t_k$)")
        logging.info(
                "& --- " * (nb_pts_time - 1)
                + f"& {(numerator.mean()/denominator.mean()).sqrt().item():.2E} \\\\"
        )

    @staticmethod
    def nth_derivatives(order, y, x):
        """
        calculate the derivatives of y wrt x with order `order`
        """
        for cur_dim, cur_order in enumerate(order):
            for _ in range(int(cur_order)):
                try:
                    grads = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
                except RuntimeError as e:
                    # when very high order derivatives are taken for polynomial function
                    # it has 0 gradient but torch has difficulty knowing that
                    # hence we handle such error separately
                    # logging.debug(e)
                    return torch.zeros_like(y)

                # update y
                y = grads[cur_dim]
        return y

    def adjusted_phi(self, x, T, coordinate, patch):
        """
        find the suitable terminal condition based on the value of patch
        when branch_patches=1, this function always outputs self.phi_fun(x)
        """
        if patch == 0:
            return self.phi_fun(x, coordinate)
        else:
            # TODO: account for coordinate in this part of code..
            self.eval()
            xx = torch.stack((T.reshape(-1), x.reshape(-1)), dim=-1)
            self.train()
            return self(xx, patch=patch - 1).reshape(-1, self.nb_path_per_state)

    def code_to_function(self, code, x, T, coordinate, patch=0):
        """
        calculate the functional of tree based on code and x

        there are two ways of representing the code
        1. negative code of size d
                (neg_num_1, ..., neg_num_d) -> d/dx1^{-neg_num_1 - 1} ... d/dxd^{-neg_num_d - 1} phi(x1, ..., xd)
        2. positive code of size n
                (pos_num_1, ..., pos_num_n) -> d/dy1^{pos_num_1 - 1} ... d/dyd^{-pos_num_1 - 1} phi(y1, ..., yn)
                    y_i is the derivatives of phi wrt x with order self.deriv_map[i-1]

        shape of x      -> d x batch
        shape of output -> batch
        """
        x = x.detach().clone().requires_grad_(True)
        fun_val = torch.zeros_like(x[0])

        # negative code of size d
        if code[0] < 0:
            return self.nth_derivatives(
                -code - 1, self.adjusted_phi(x, T, coordinate, patch), x
            ).detach()

        # positive code of size d
        if code[0] > 0:
            y = []
            for idx, order in enumerate(self.deriv_map):
                if self.zeta_map[idx] < 0:
                    y.append(
                        self.nth_derivatives(
                            order, self(x.T, p_or_u="p", patch=patch), x
                        )
                    )
                else:
                    y.append(
                        self.nth_derivatives(
                            order, self.adjusted_phi(x, T, self.zeta_map[idx], patch), x
                        ).detach()
                    )
            y = torch.stack(y[: self.n]).requires_grad_()

            return self.nth_derivatives(
                code - 1, self.f_fun(y, coordinate), y
            ).detach()

        return fun_val.detach()

    def gen_bm(self, dt, nb_states, var=None):
        """
        generate brownian motion sqrt{dt} x Gaussian

        when self.antithetic=true, we generate
        dw = sqrt{dt} x Gaussian of size nb_states//2
        and return (dw, -dw)
        """
        var = 2 * self.beta if var is None else var
        if not torch.is_tensor(dt):
            dt = torch.tensor(dt)
        dt = dt.clip(min=0.0)  # so that we can safely take square root of dt

        if self.antithetic:
            # antithetic variates
            normal = torch.randn(
                self.dim, nb_states, self.nb_path_per_state // 2, device=self.device
            ).repeat(1, 1, 2)
            normal[:, :, : (self.nb_path_per_state // 2)] *= -1
        else:
            # usual generation
            normal = torch.randn(
                self.dim, nb_states, self.nb_path_per_state, device=self.device
            )
        return torch.sqrt(var * dt) * normal

    def helper_negative_code_on_f(self, t, T, x, mask, H, code, patch, coordinate):
        ans = torch.zeros_like(t)
        order = tuple(-code - 1)
        # if c is not in the lookup, add it
        if order not in self.fdb_lookup.keys():
            start = time.time()
            self.fdb_lookup[order] = fdb_nd(self.n, order)
            self.fdb_runtime += time.time() - start
        L = self.fdb_lookup[order]
        unif = torch.rand(t.shape[0], self.nb_path_per_state, device=self.device)
        idx = (unif * len(L)).long()
        idx_counter = 0
        for fdb in self.fdb_lookup[order]:
            mask_tmp = mask * (idx == idx_counter)
            if mask_tmp.any():
                A = self.gen_sample_batch(
                    t,
                    T,
                    x,
                    mask_tmp,
                    fdb.coeff * len(L) * H,
                    np.array(fdb.lamb) + 1,
                    patch,
                    coordinate,
                )
                for ll, k_arr in fdb.l_and_k.items():
                    for q in range(self.n):
                        for _ in range(k_arr[q]):
                            A = A * self.gen_sample_batch(
                                t,
                                T,
                                x,
                                mask_tmp,
                                torch.ones_like(t),
                                -self.deriv_map[q] - ll - 1,
                                patch,
                                self.zeta_map[q],
                            )
                ans = ans.where(~mask_tmp, A)
            idx_counter += 1
        return ans

    def gen_sample_batch(self, t, T, x, mask, H, code, patch, coordinate):
        """
        recursive function to calculate E[ H(t, x, code) ]

        t    -> current time
             -> shape of nb_states x nb_paths_per_state
        T    -> terminal time
             -> shape of nb_states x nb_paths_per_state
        x    -> value of brownian motion at time t
             -> shape of d x nb_states x nb_paths_per_state
        mask -> mask[idx]=1 means the state at index idx is still alive
             -> mask[idx]=0 means the state at index idx is dead
             -> shape of nb_states x nb_paths_per_state
        H    -> cummulative value of the product of functional H
             -> shape of nb_states x nb_paths_per_state
        code -> determine the operation to be taken on the functions f and phi
             -> negative code of size d or positive code of size n
        """
        # return zero tensor when no operation is needed
        ans = torch.zeros_like(t)
        if ~mask.any():
            return ans

        nb_states, _ = t.shape

        if coordinate < 0:
            unif = (
                torch.rand(nb_states * self.nb_path_per_state, device=self.device)
                     .reshape(nb_states, self.nb_path_per_state)
            )
            tau = self.tau_lo + (self.tau_hi - self.tau_lo) * unif
            dw = self.gen_bm(tau, nb_states, var=1)
            unif = torch.rand(nb_states, self.nb_path_per_state, device=self.device)
            order = -code - 1
            L = [fdb for fdb in fdb_nd(2, order) if max(fdb.lamb) < 2]
            idx_counter = 0
            idx = (unif * len(L) * self.dim ** 2).long()
            if coordinate == -2:
                idx *= (self.dim + 2)
            for i in range(self.dim):
                for j in range(self.dim):
                    for fdb in L:
                        if coordinate == -1:
                            # coordinate -1 -> apply code to p
                            mask_tmp = mask.bool() * (idx == idx_counter)
                            if mask_tmp.any():
                                A = (
                                        H * fdb.coeff
                                        * len(L) * self.dim ** 2
                                        * self.dim ** 2
                                        * (dw ** 2).sum(dim=0)
                                        * (self.tau_hi - self.tau_lo)
                                        / (2 * tau)
                                )
                                if self.dim > 2:
                                    A = A / (self.dim - 2)
                                elif self.dim == 2:
                                    A = -A * torch.log((dw ** 2).sum(dim=0).sqrt())
                                code_increment = np.zeros_like(code)
                                code_increment[j] += 1
                                if fdb.lamb[0] == 0:
                                    A = A * self.gen_sample_batch(
                                        t,
                                        T,
                                        x + dw,
                                        mask_tmp,
                                        torch.ones_like(t),
                                        -code_increment - 1,
                                        patch,
                                        i,
                                    )
                                code_increment[j] -= 1
                                code_increment[i] += 1
                                if fdb.lamb[1] == 0:
                                    A = A * self.gen_sample_batch(
                                        t,
                                        T,
                                        x + dw,
                                        mask_tmp,
                                        torch.ones_like(t),
                                        -code_increment - 1,
                                        patch,
                                        j,
                                    )

                                for ll, k_arr in fdb.l_and_k.items():
                                    for _ in range(k_arr[1]):
                                        A = A * self.gen_sample_batch(
                                            t,
                                            T,
                                            x + dw,
                                            mask_tmp,
                                            torch.ones_like(t),
                                            -code_increment - ll - 1,
                                            patch,
                                            j,
                                        )
                                    code_increment[i] -= 1
                                    code_increment[j] += 1
                                    for _ in range(k_arr[0]):
                                        A = A * self.gen_sample_batch(
                                            t,
                                            T,
                                            x + dw,
                                            mask_tmp,
                                            torch.ones_like(t),
                                            -code_increment - ll - 1,
                                            patch,
                                            i,
                                        )
                                ans = ans.where(~mask_tmp, A)
                            idx_counter += 1

                        elif coordinate == -2:
                            # coordinate -2 -> apply code to \partial_t p + beta * \Delta p
                            for k in range(self.dim + 2):
                                mask_tmp = mask.bool() * (idx == idx_counter)
                                if mask_tmp.any():
                                    A = (
                                            H * fdb.coeff
                                            * len(L) * self.dim ** 2 * (self.dim + 2)
                                            * self.dim ** 2
                                            * (dw ** 2).sum(dim=0)
                                            * (self.tau_hi - self.tau_lo)
                                            / (2 * tau)
                                    )
                                    if self.dim > 2:
                                        A = A / (self.dim - 2)
                                    elif self.dim == 2:
                                        A = -A * torch.log((dw ** 2).sum(dim=0).sqrt())
                                    code_increment = np.zeros_like(code)
                                    if k < self.dim:
                                        A = 2 * self.beta * A
                                        code_increment[k] += 1
                                    elif k == self.dim + 1:
                                        # the only difference between the last two k is the indexing of i, j
                                        i, j = j, i
                                    code_increment[j] += 1
                                    if fdb.lamb[0] == 0:
                                        A = A * self.gen_sample_batch(
                                            t,
                                            T,
                                            x + dw,
                                            mask_tmp,
                                            torch.ones_like(t),
                                            -code_increment - 1,
                                            patch,
                                            i,
                                        )
                                    code_increment[j] -= 1
                                    code_increment[i] += 1
                                    if fdb.lamb[1] == 0:
                                        if k < self.dim:
                                            A = A * self.gen_sample_batch(
                                                t,
                                                T,
                                                x + dw,
                                                mask_tmp,
                                                torch.ones_like(t),
                                                -code_increment - 1,
                                                patch,
                                                j,
                                            )
                                        else:
                                            A = A * self.helper_negative_code_on_f(
                                                t,
                                                T,
                                                x + dw,
                                                mask_tmp,
                                                -torch.ones_like(t),
                                                -code_increment - 1,
                                                patch,
                                                j,
                                            )

                                    for ll, k_arr in fdb.l_and_k.items():
                                        if k < self.dim:
                                            for _ in range(k_arr[1]):
                                                A = A * self.gen_sample_batch(
                                                    t,
                                                    T,
                                                    x + dw,
                                                    mask_tmp,
                                                    torch.ones_like(t),
                                                    -code_increment - ll - 1,
                                                    patch,
                                                    j,
                                                )
                                        else:
                                            for _ in range(k_arr[1]):
                                                A = A * self.helper_negative_code_on_f(
                                                    t,
                                                    T,
                                                    x + dw,
                                                    mask_tmp,
                                                    -torch.ones_like(t),
                                                    -code_increment - ll - 1,
                                                    patch,
                                                    j,
                                                )
                                        code_increment[i] -= 1
                                        code_increment[j] += 1
                                        for _ in range(k_arr[0]):
                                            A = A * self.gen_sample_batch(
                                                t,
                                                T,
                                                x + dw,
                                                mask_tmp,
                                                torch.ones_like(t),
                                                -code_increment - ll - 1,
                                                patch,
                                                i,
                                            )
                                    ans = ans.where(~mask_tmp, A)
                                idx_counter += 1
            return ans

        tau = Exponential(
            self.exponential_lambda
            * torch.ones(nb_states, self.nb_path_per_state, device=self.device)
        ).sample()
        dw = self.gen_bm(T - t, nb_states)

        ############################### for t + tau >= T
        mask_now = mask.bool() * (t + tau >= T)
        if mask_now.any():
            tmp = (
                H[mask_now]
                * self.code_to_function(
                    code, (x + dw)[:, mask_now], T[mask_now], coordinate, patch
                )
                / torch.exp(-self.exponential_lambda * (T - t)[mask_now])
            )
            ans[mask_now] = tmp

        ############################### for t + tau < T
        dw = self.gen_bm(tau, nb_states)
        mask_now = mask.bool() * (t + tau < T)

        # return when all processes die
        if ~mask_now.any():
            return ans

        # uniform distribution to choose from the set of mechanism
        unif = torch.rand(nb_states, self.nb_path_per_state, device=self.device)

        # identity code (-1, ..., -1) of size d
        if (len(code) == self.dim) and (code == [-1] * self.dim).all():
            if mask_now.any():
                tmp = self.gen_sample_batch(
                    t + tau,
                    T,
                    x + dw,
                    mask_now,
                    H
                    / self.exponential_lambda
                    / torch.exp(-self.exponential_lambda * tau),
                    np.array([1] * self.n),
                    patch,
                    coordinate,
                )
                ans = ans.where(~mask_now, tmp)

        # negative code of size d
        elif code[0] < 0:
            order = tuple(-code - 1)
            # if c is not in the lookup, add it
            if order not in self.fdb_lookup.keys():
                start = time.time()
                self.fdb_lookup[order] = fdb_nd(self.n, order)
                self.fdb_runtime += time.time() - start
            L = self.fdb_lookup[order]
            idx = (unif * len(L)).long()
            idx_counter = 0

            # loop through all fdb elements
            for fdb in L:
                mask_tmp = mask_now * (idx == idx_counter)
                if mask_tmp.any():
                    A = fdb.coeff * self.gen_sample_batch(
                        t + tau,
                        T,
                        x + dw,
                        mask_tmp,
                        len(L)
                        * H
                        / self.exponential_lambda
                        / torch.exp(-self.exponential_lambda * tau),
                        np.array(fdb.lamb) + 1,
                        patch,
                        coordinate,
                    )

                    for ll, k_arr in fdb.l_and_k.items():
                        for q in range(self.n):
                            # for u
                            for _ in range(k_arr[q]):
                                A = A * self.gen_sample_batch(
                                    t + tau,
                                    T,
                                    x + dw,
                                    mask_tmp,
                                    torch.ones_like(t),
                                    -self.deriv_map[q] - ll - 1,
                                    patch,
                                    self.zeta_map[q],
                                )
                    ans = ans.where(~mask_tmp, A)
                idx_counter += 1

        # positive code of size n
        elif code[0] > 0:
            idx = (unif * self.mechanism_tot_len).long()
            idx_counter = 0

            # positive code part 1
            for i in range(self.n):
                for j in range(self.n):
                    for k in range(self.dim):
                        mask_tmp = mask_now * (idx == idx_counter)
                        if mask_tmp.any():
                            code_increment = np.zeros_like(self.deriv_map[i])
                            code_increment[k] += 1
                            A = self.gen_sample_batch(
                                t + tau,
                                T,
                                x + dw,
                                mask_tmp,
                                torch.ones_like(t),
                                -self.deriv_map[i] - code_increment - 1,
                                patch,
                                self.zeta_map[i],
                            )
                            B = self.gen_sample_batch(
                                t + tau,
                                T,
                                x + dw,
                                mask_tmp,
                                torch.ones_like(t),
                                -self.deriv_map[j] - code_increment - 1,
                                patch,
                                self.zeta_map[i],
                            )
                            # only code + 1 in the dimension j and l
                            code_increment = np.zeros_like(code)
                            code_increment[i] += 1
                            code_increment[j] += 1
                            tmp = self.gen_sample_batch(
                                t + tau,
                                T,
                                x + dw,
                                mask_tmp,
                                -self.beta
                                * self.mechanism_tot_len
                                * A
                                * B
                                * H
                                / self.exponential_lambda
                                / torch.exp(-self.exponential_lambda * tau),
                                code + code_increment,
                                patch,
                                coordinate,
                            )
                            ans = ans.where(~mask_tmp, tmp)
                        idx_counter += 1

            # positive code part 2
            for k in range(self.nprime, self.n):
                for fdb in self.fdb_lookup[tuple(self.deriv_map[k])]:
                    mask_tmp = mask_now * (idx == idx_counter)
                    if mask_tmp.any():
                        A = fdb.coeff * self.gen_sample_batch(
                            t + tau,
                            T,
                            x + dw,
                            mask_tmp,
                            torch.ones_like(t),
                            np.array(fdb.lamb) + 1,
                            patch,
                            self.zeta_map[k],
                        )
                        for ll, k_arr in fdb.l_and_k.items():
                            for q in range(self.n):
                                for _ in range(k_arr[q]):
                                    A = A * self.gen_sample_batch(
                                        t + tau,
                                        T,
                                        x + dw,
                                        mask_tmp,
                                        torch.ones_like(t),
                                        -self.deriv_map[q] - ll - 1,
                                        patch,
                                        self.zeta_map[q],
                                    )
                        code_increment = np.zeros_like(code)
                        code_increment[k] += 1
                        tmp = self.gen_sample_batch(
                            t + tau,
                            T,
                            x + dw,
                            mask_tmp,
                            self.mechanism_tot_len
                            * A
                            * H
                            / self.exponential_lambda
                            / torch.exp(-self.exponential_lambda * tau),
                            code + code_increment,
                            patch,
                            coordinate,
                        )
                        ans = ans.where(~mask_tmp, tmp)
                    idx_counter += 1

            # positive code part 3
            for k in range(self.nprime):
                mask_tmp = mask_now * (idx == idx_counter)
                if mask_tmp.any():
                    code_increment = np.zeros_like(code)
                    code_increment[k] += 1
                    A = -self.gen_sample_batch(
                        t + tau,
                        T,
                        x + dw,
                        mask_tmp,
                        self.mechanism_tot_len
                        * H
                        / self.exponential_lambda
                        / torch.exp(-self.exponential_lambda * tau),
                        code + code_increment,
                        patch,
                        coordinate,
                    )
                    A = A * self.gen_sample_batch(
                        t + tau,
                        T,
                        x + dw,
                        mask_tmp,
                        torch.ones_like(t),
                        -self.deriv_map[k] - 1,
                        patch,
                        -2,
                    )
                    ans = ans.where(~mask_tmp, A)
                idx_counter += 1

        return ans

    def gen_sample(
        self, patch, coordinate=None, code=None, t=None, discard_outlier=True, gen_y=True,
    ):
        """
        generate sample based on the (t, x) boundary and the function gen_sample_batch
        """
        code = np.array([[-1] * self.dim]) if code is None else code
        coordinate = np.array([0]) if coordinate is None else coordinate
        if t is None:
            nb_states = self.nb_states
        else:
            nb_states, _ = t.shape
        states_per_batch = min(nb_states, self.nb_states_per_batch)
        batches = math.ceil(nb_states / states_per_batch)
        t_lo, t_hi = self.adjusted_t_boundaries[patch]
        t_hi = min(self.t_hi, t_hi)
        x_lo, x_hi = self.adjusted_x_boundaries
        xx, yy = [], []
        for batch_now in range(batches):
            start = time.time()
            unif = (
                torch.rand(states_per_batch, device=self.device)
                .repeat(self.nb_path_per_state)
                .reshape(self.nb_path_per_state, -1)
                .T
            )
            t = t_lo + (t_hi - t_lo) * unif
            unif = (
                torch.rand(self.dim * states_per_batch, device=self.device)
                .repeat(self.nb_path_per_state)
                .reshape(self.nb_path_per_state, -1)
                .T.reshape(self.dim, states_per_batch, self.nb_path_per_state)
            )
            x = x_lo + (x_hi - x_lo) * unif
            # fix all dimensions (except the first) to be the middle value
            if self.dim > 1 and self.fix_all_dim_except_first:
                x[1:, :, :] = (x_hi + x_lo) / 2
            T = (t_lo + self.delta_t) * torch.ones_like(t)
            xx.append(torch.cat((t[:, :1], x[:, :, 0].T), dim=-1).detach())
            if gen_y:
                yyy = []
                for (idx, c) in zip(coordinate, code):
                    yy_tmp = self.gen_sample_batch(
                        t,
                        T,
                        x,
                        torch.ones_like(t),
                        torch.ones_like(t),
                        c,
                        patch,
                        idx,
                    ).detach()
                    if discard_outlier:
                        # let (lo, hi) be
                        # (self.outlier_percentile, 100 - self.outlier_percentile)
                        # percentile of yy_tmp
                        #
                        # set the boundary as [lo-1000*(hi-lo), hi+1000*(hi-lo)]
                        # samples out of this boundary is considered as outlier and removed
                        lo, hi = (
                            yy_tmp.nanquantile(
                                self.outlier_percentile / 100, dim=1, keepdim=True
                            ),
                            yy_tmp.nanquantile(
                                1 - self.outlier_percentile / 100, dim=1, keepdim=True
                            ),
                        )
                        lo, hi = lo - self.outlier_multiplier * (hi - lo), hi + self.outlier_multiplier * (hi - lo)
                        mask = torch.logical_and(lo <= yy_tmp, yy_tmp <= hi)
                    else:
                        mask = ~yy_tmp.isnan()
                    # plt.plot(yy_tmp[0, :].detach().cpu(), '+')
                    # plt.title("Before discarding outliers.")
                    # plt.show()
                    # plt.plot((yy_tmp * mask)[0, :].detach().cpu(), '+')
                    # plt.title("After discarding outliers.")
                    # plt.show()
                    yyy.append((yy_tmp.nan_to_num() * mask).sum(dim=1) / mask.sum(dim=1))
                yy.append(torch.stack(yyy, dim=-1))
                logging.info(f"Generated {batch_now + 1} out of {batches} batches with {time.time() - start} seconds.")

        return (
            torch.cat(xx, dim=0),
            torch.cat(yy, dim=0) if yy else None,
        )

    def gen_sample_for_p(self, gen_y=True, overtrain_rate=.5):
        states_per_batch = min(self.nb_states, self.nb_states_per_batch)
        batches = math.ceil(self.nb_states / states_per_batch)
        xx, yy = [], []
        # widen the domain from [x_lo, x_hi] to [x_lo - .5*(x_hi-x_lo), x_hi + .5*(x_hi-x_lo)]
        x_lo, x_hi = self.x_lo, self.x_hi
        x_lo, x_hi = x_lo - overtrain_rate * (x_hi - x_lo), x_hi + overtrain_rate * (x_hi - x_lo)
        for _ in range(batches):
            unif = (
                torch.rand(self.dim * states_per_batch, device=self.device)
                     .reshape(states_per_batch, self.dim)
            )
            x = (x_lo + (x_hi - x_lo) * unif).T
            if self.dim > 1 and self.fix_all_dim_except_first:
                x[1:, :] = (x_hi + x_lo) / 2
            if gen_y:
                if self.quantization:
                    y = self.calculate_p_from_u_quant(x.T)
                else:
                    y = self.calculate_p_from_u(x.T)
                yy.append(y)
            xx.append(x)
        return (
            torch.cat(xx, dim=-1),
            torch.cat(yy, dim=-1) if yy else None,
        )

    def plot_u(self, epoch, x=None, y=None, save_dir=None):
        self.eval()
        grid = np.linspace(self.x_lo, self.x_hi, 100)
        x_mid = x[0, 2].item() if self.fix_all_dim_except_first else (self.x_lo + self.x_hi) / 2
        t_lo = x[0, 0].item() if self.fix_all_dim_except_first else self.T / 2
        grid_nd = np.concatenate(
            (
                t_lo * np.ones((1, 100)),
                np.expand_dims(grid, axis=0),
                x_mid * np.ones((self.dim - 1, 100)),
            ),
            axis=0,
        )
        nn = (
            self(
                torch.tensor(
                    grid_nd.T,
                    device=self.device,
                    dtype=torch.get_default_dtype()
                ), patch=0
            ).detach().cpu()
        )
        for i in range(self.dim):
            exact = (
                self.exact_u_fun(
                    torch.tensor(
                        grid_nd.T,
                        device=self.device,
                        dtype=torch.get_default_dtype()), i
                ).detach().cpu()
            )
            fig = plt.figure()
            if self.fix_all_dim_except_first:
                plt.plot(x.detach().cpu()[:, 1], y.detach().cpu()[:, i], '+', label="MC samples")
            plt.plot(grid, nn[:, i], label=f"NN")
            plt.plot(grid, exact, label=f"exact")
            if self.plot_y_lim is not None and epoch == (self.epochs - 1):
                plt.ylim(self.plot_y_lim[i])
            plt.title(f"Epoch {epoch:04}")
            plt.legend(loc="upper left")
            if save_dir is None:
                fig.savefig(
                    f"{self.working_dir}/plot/u{i}/epoch_{epoch:04}.png", bbox_inches="tight"
                )
            else:
                fig.savefig(
                    f"{save_dir}/plot/u{i}/epoch_{epoch:04}.png", bbox_inches="tight"
                )
            plt.close()

    def load_dict(self, path_to_model):
        self.loaded_dict = True
        self.load_state_dict(torch.load(path_to_model))

    def train_and_eval(self, debug_mode=False):
        """
        generate sample and evaluate (plot) NN approximation when debug_mode=True
        """
        for p in range(self.patches):
            if not (self.debug or self.loaded_dict):
                # do not need to train for p network in debug mode
                # initialize optimizer
                optimizer = torch.optim.Adam(
                    self.parameters(), lr=self.lr, weight_decay=self.weight_decay
                )
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=self.lr_milestones,
                    gamma=self.lr_gamma,
                )
                train_p_start = time.time()
                for epoch in range(self.epochs):
                    start = time.time()
                    if epoch % 100 == 0:  # only generate in the beginning
                        x, y = self.gen_sample_for_p()
                        poisson_rhs = 0
                        order = np.array([0] * self.dim)
                        xx = x.detach().clone().requires_grad_(True)
                        for i in range(self.dim):
                            for j in range(self.dim):
                                order[i] += 1
                                tmp = self.nth_derivatives(
                                    order, self.phi_fun(xx, j), xx
                                )
                                order[i] -= 1
                                order[j] += 1
                                tmp *= self.nth_derivatives(
                                    order, self.phi_fun(xx, i), xx
                                )
                                order[j] -= 1
                                poisson_rhs -= tmp
                        poisson_rhs = poisson_rhs.detach()

                    optimizer.zero_grad()
                    self.train()
                    loss = self.loss(self(x.T, p_or_u="p", patch=p).squeeze(), y)
                    self.eval()
                    poisson_lhs = 0
                    xx = x.detach().clone().requires_grad_(True)
                    for i in range(self.dim):
                        order[i] += 2
                        poisson_lhs += self.nth_derivatives(
                            order, self(xx.T, p_or_u="p", patch=p), xx
                        )
                        order[i] -= 2
                    loss += self.loss(poisson_lhs, poisson_rhs)

                    # update model weights and schedule
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.eval()
                    grid = np.linspace(self.x_lo, self.x_hi, 100)
                    x_mid = x[1, 0].item() if self.fix_all_dim_except_first else (self.x_lo + self.x_hi) / 2
                    t_lo = self.T
                    grid_nd = np.concatenate(
                        (
                            t_lo * np.ones((1, 100)),
                            np.expand_dims(grid, axis=0),
                            x_mid * np.ones((self.dim - 1, 100)),
                        ),
                        axis=0,
                    )
                    nn = (
                        self(
                            torch.tensor(
                                grid_nd[1:].T,
                                device=self.device,
                                dtype=torch.get_default_dtype()
                            ), patch=0, p_or_u="p"
                        ).detach().cpu()
                    )
                    exact = (
                        self.exact_p_fun(
                            torch.tensor(
                                grid_nd.T,
                                device=self.device,
                                dtype=torch.get_default_dtype()
                            )
                        ).detach().cpu()
                    )
                    if not self.fix_all_dim_except_first:
                        exact += nn.mean() - exact.mean()
                    fig = plt.figure()
                    if self.fix_all_dim_except_first:
                        plt.plot(x.detach().cpu()[0, :], y.detach().cpu(), '+', label="MC samples")
                    plt.plot(grid, nn, label=f"NN")
                    plt.plot(grid, exact, label=f"exact")
                    plt.title(f"Epoch {epoch:04}")
                    plt.legend(loc="upper left")
                    fig.savefig(
                        f"{self.working_dir}/plot/p/epoch_{epoch:04}.png", bbox_inches="tight"
                    )
                    plt.close()
                    torch.save(
                        self.state_dict(), f"{self.working_dir}/model/epoch_{epoch:04}.pt"
                    )
                    logging.info(
                        f"Pre-training epoch {epoch:3.0f}: one loop takes {time.time() - start:4.0f} seconds with loss {loss.detach():.2E}."
                    )
                logging.info(
                    f"Training of p takes {time.time() - train_p_start:4.0f} seconds."
                )

            # initialize optimizer
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.lr_milestones,
                gamma=self.lr_gamma,
            )
            train_u_start = time.time()
            # loop through epochs
            for epoch in range(self.epochs):
                # clear gradients and evaluate training loss
                optimizer.zero_grad()
                if epoch % self.epochs == 0:  # only generate in the beginning
                    # the running statistics for p is very different from time to time, so we do not track them
                    self.eval()
                    start = time.time()
                    logging.info(f"Generating new samples at epoch {epoch:3.0f}.")
                    x, y = self.gen_sample(
                        patch=p,
                        coordinate=np.array(range(self.dim)),
                        code=-np.ones((self.dim, self.dim), dtype=int),
                        discard_outlier=True,
                    )
                    logging.info(
                        f"Sample generation takes {time.time() - start:4.0f} seconds."
                    )
                self.train()
                start = time.time()
                predict = self(x, patch=p)
                loss = self.loss(y, predict)
                if self.deriv_condition_coeff > 0:
                    self.eval()
                    xx = x.T.detach().clone().requires_grad_(True)
                    grad = 0
                    for (idx, c) in zip(self.deriv_condition_zeta_map, self.deriv_condition_deriv_map):
                        # additional t coordinate
                        grad += self.nth_derivatives(
                            np.insert(c, 0, 0), self(xx.T, patch=p)[:, idx], xx
                        )
                    loss += self.deriv_condition_coeff * self.loss(grad, torch.zeros_like(grad))

                # additional loss regarding poisson equation
                if self.poisson_loss_coeff > 0:
                    poisson_lhs, poisson_rhs = 0, 0
                    order = np.array([0] * (self.dim + 1))
                    for i in range(self.dim):
                        order[i + 1] += 2
                        poisson_lhs += self.nth_derivatives(
                            order, self(xx.T, p_or_u="p", patch=p), xx
                        )
                        order[i + 1] -= 2
                    for i in range(self.dim):
                        for j in range(self.dim):
                            order[i + 1] += 1
                            tmp = self.nth_derivatives(
                                order, self(xx.T, patch=p)[:, j], xx
                            )
                            order[i + 1] -= 1
                            order[j + 1] += 1
                            tmp *= self.nth_derivatives(
                                order, self(xx.T, patch=p)[:, i], xx
                            )
                            order[j + 1] -= 1
                            poisson_rhs -= tmp
                    loss += self.poisson_loss_coeff * self.loss(poisson_lhs, poisson_rhs)

                # update model weights and schedule
                loss.backward()
                optimizer.step()
                scheduler.step()

                self.plot_u(epoch, x, y)
                torch.save(
                    self.state_dict(), f"{self.working_dir}/model/epoch_{epoch:04}.pt"
                )

                logging.info(
                    f"Epoch {epoch:3.0f}: one loop takes {time.time() - start:4.0f} seconds with loss {loss.detach():.2E}."
                )
            logging.info(
                f"Training of u takes {time.time() - train_u_start:4.0f} seconds."
            )


if __name__ == "__main__":
    # configurations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    problem = ["taylor_green_2d", "abc_3d"][0]

    if problem == "taylor_green_2d":
        # taylor green vortex
        T, x_lo, x_hi, beta = 0.25, 0, 2 * math.pi, 1.0
        # deriv_map is n x d array defining lambda_1, ..., lambda_n
        deriv_map = np.array(
            [
                [1, 0],  # for nabla p
                [0, 1],
                [0, 0],  # for u
                [0, 0],
                [1, 0],  # for nabla u1
                [0, 1],
                [1, 0],  # for nabla u2
                [0, 1],
            ]
        )
        zeta_map = np.array([-1, -1, 0, 1, 0, 0, 1, 1])
        deriv_condition_deriv_map = np.array(
            [
                [1, 0],
                [0, 1],
            ]
        )
        deriv_condition_zeta_map = np.array([0, 1])
    elif problem == "abc_3d":
        # ABC flow
        T, x_lo, x_hi, beta = 0.7, 0, 2 * math.pi, 0.01
        A = B = C = 0.5
        # deriv_map is n x d array defining lambda_1, ..., lambda_n
        deriv_map = np.array(
            [
                [1, 0, 0],  # for nabla p
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 0],  # for u
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0],  # for nabla u1
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],  # for nabla u2
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],  # for nabla u3
                [0, 1, 0],
                [0, 0, 1],
            ]
        )
        zeta_map = np.array([-1, -1, -1, 0, 1, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2])
        deriv_condition_deriv_map = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )
        deriv_condition_zeta_map = np.array([0, 1, 2])

    _, dim = deriv_map.shape

    def f_fun(y, i):
        """
        TODO: some descriptions about deriv_map...
        """
        f = -y[i]
        for j in range(dim):
            f += -y[dim + j] * y[2 * dim + dim * i + j]
        return f

    def g_fun(x, i):
        if problem == "taylor_green_2d":
            # taylor green vortex
            if i == 0:
                return -torch.cos(x[0]) * torch.sin(x[1])
            else:
                return torch.sin(x[0]) * torch.cos(x[1])
        elif problem == "abc_3d":
            # ABC flow
            if i == 0:
                return A * torch.sin(x[2]) + C * torch.cos(x[1])
            elif i == 1:
                return B * torch.sin(x[0]) + A * torch.cos(x[2])
            else:
                return C * torch.sin(x[1]) + B * torch.cos(x[0])

    def exact_u_fun(x, i):
        if problem == "taylor_green_2d":
            # taylor green vortex
            if i == 0:
                return -torch.cos(x[:, 1]) * torch.sin(x[:, 2]) * torch.exp(-2 * beta * (T - x[:, 0]))
            else:
                return torch.sin(x[:, 1]) * torch.cos(x[:, 2]) * torch.exp(-2 * beta * (T - x[:, 0]))
        elif problem == "abc_3d":
            # ABC flow
            if i == 0:
                return (A * torch.sin(x[:, 3]) + C * torch.cos(x[:, 2])) * torch.exp(-beta * (T - x[:, 0]))
            elif i == 1:
                return (B * torch.sin(x[:, 1]) + A * torch.cos(x[:, 3])) * torch.exp(-beta * (T - x[:, 0]))
            else:
                return (C * torch.sin(x[:, 2]) + B * torch.cos(x[:, 1])) * torch.exp(-beta * (T - x[:, 0]))

    def exact_p_fun(x):
        if problem == "taylor_green_2d":
            # taylor green vortex
            return (
                -1
                / 4
                * torch.exp(-4 * beta * (T - x[:, 0]))
                * (torch.cos(2 * x[:, 1]) + torch.cos(2 * x[:, 2]))
            )
        elif problem == "abc_3d":
            # ABC flow
            return -torch.exp(-2 * beta * (T - x[:, 0])) * (
                A * C * torch.sin(x[:, 3]) * torch.cos(x[:, 2])
                + B * A * torch.sin(x[:, 1]) * torch.cos(x[:, 3])
                + C * B * torch.sin(x[:, 2]) * torch.cos(x[:, 1])
            )

    # initialize model and training
    model = Net(
        deriv_map=deriv_map,
        zeta_map=zeta_map,
        deriv_condition_deriv_map=deriv_condition_deriv_map,
        deriv_condition_zeta_map=deriv_condition_zeta_map,
        f_fun=f_fun,
        phi_fun=g_fun,
        exact_p_fun=exact_p_fun,
        exact_u_fun=exact_u_fun,
        T=T,
        t_lo=0.0,
        t_hi=T,
        x_lo=x_lo,
        x_hi=x_hi,
        device=device,
        verbose=True,
        epochs=3000,
        branch_lr=1e-2,
        lr_milestones=[1000, 2000],
        lr_gamma=.1,
        branch_nb_path_per_state=1000,
        branch_nb_states=100000,
        branch_nb_states_per_batch=1000,
        beta=beta,
        layers=2,
        neurons=20,
        batch_normalization=True,
        debug=False,
        branch_activation="tanh",
        poisson_loss_coeff=0.,
        deriv_condition_coeff=1.,
        quantization=False,
        fix_all_dim_except_first=False,
        plot_y_lim=[[-1, 1], [-1, 1]] if problem == "taylor_green_2d" else [[-1, 0], [-1.05, 0.05], [-.55, .55]],
    )
    # model.error_calculation("logs/20220411-134639/model/epoch_2999.pt")
    # model.error_calculation("logs/20220411-114948/model/epoch_2999.pt", nb_pts_spatial=2*45+1)
    model.train_and_eval()
