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
        poisson_loss=False,
        overtrain_rate=0.1,
        device="cpu",
        branch_activation="tanh",
        verbose=False,
        fix_all_dim_except_first=False,
        branch_patches=1,
        outlier_percentile=1,
        outlier_multiplier=10,
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
                    [torch.nn.Linear(self.dim + 1, neurons, device=device)]
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
                    [torch.nn.BatchNorm1d(self.dim + 1, device=device)]
                    + [
                        torch.nn.BatchNorm1d(neurons, device=device)
                        for _ in range(layers + 1)
                    ]
                )
                for _ in range(branch_patches)
            ]
        )
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
        self.t_hi = t_hi
        self.T = T
        self.beta = beta
        self.delta_t = (T - t_lo) / branch_patches
        self.outlier_percentile = outlier_percentile
        self.outlier_multiplier = outlier_multiplier

        self.exponential_lambda = (
            branch_exponential_lambda
            if branch_exponential_lambda is not None
            else -math.log(0.8) / T
        )
        self.epochs = epochs
        self.antithetic = antithetic
        self.poisson_loss = poisson_loss
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

    def forward(self, x, patch=None, p_or_u="u"):
        """
        self(x) evaluates the neural network approximation NN(x)
        """
        layer = self.u_layer if p_or_u == "u" else self.p_layer
        bn_layer = self.u_bn_layer if p_or_u == "u" else self.p_bn_layer

        if self.debug:
            # return the exact function for debug purposes
            if p_or_u == "p":
                return self.exact_p_fun(x)
            else:
                return torch.stack([self.exact_u_fun(x, i) for i in range(self.dim)], dim=-1)

        # normalization to make sure x is roughly within the range of [0, 1] x (dim + 1)
        x_lo = torch.tensor([self.t_lo] + [self.x_lo] * self.dim, device=self.device)
        x_hi = torch.tensor([self.T] + [self.x_hi] * self.dim, device=self.device)
        x = (x - x_lo) / (x_hi - x_lo)

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
    def nth_derivatives(order, y, x):
        """
        calculate the derivatives of y wrt x with order `order`
        """
        for cur_dim, cur_order in enumerate(order):
            for _ in range(int(cur_order)):
                try:
                    grads = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
                except RuntimeError:
                    # when very high order derivatives are taken for polynomial function
                    # it has 0 gradient but torch has difficulty knowing that
                    # hence we handle such error separately
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
        tx = torch.cat((T.unsqueeze(0), x), dim=0).detach().clone().requires_grad_(True)
        fun_val = torch.zeros_like(x[0])

        if coordinate == -1:
            # coordinate -1 -> apply code to p
            order = -code - 1
            order = np.insert(order, 0, 0)  # p has additionally t coordinate
            return self.nth_derivatives(
                order, self(x.T, p_or_u="p", patch=patch), x
            )

        if coordinate == -2:
            # coordinate -2 -> apply code to \partial_t p + beta * \Delta p
            order = -code - 1
            order = np.insert(order, 0, 1)  # p has additionally t coordinate
            ans = self.nth_derivatives(
                order, self(x.T, p_or_u="p", patch=patch), x
            )
            order[0] -= 1
            for i in range(self.dim):
                order[i + 1] += 2  # Laplacian
                ans += self.beta * self.nth_derivatives(
                    order, self(x.T, p_or_u="p", patch=patch), x
                )
                order[i + 1] -= 2
            return ans

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
                    # p has additionally t coordinate
                    y.append(
                        self.nth_derivatives(
                            np.insert(order, 0, 0), self(tx.T, p_or_u="p", patch=patch), tx
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
            )

        return fun_val

    def gen_bm(self, dt, nb_states):
        """
        generate brownian motion sqrt{dt} x Gaussian

        when self.antithetic=true, we generate
        dw = sqrt{dt} x Gaussian of size nb_states//2
        and return (dw, -dw)
        """
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
        return torch.sqrt(2 * self.beta * dt) * normal

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

        if coordinate < 0:
            # coordinate -1 -> apply code to p
            # coordinate -2 -> apply code to \partial_t p + beta * \Delta p
            mask_now = mask.bool()
            tx = torch.cat((t.unsqueeze(0), x), dim=0)
            tmp = H[mask_now] * self.code_to_function(
                code, tx[:, mask_now], T[mask_now], coordinate, patch
            )
            ans[mask_now] = tmp
            return ans

        nb_states, _ = t.shape
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
                        for q in range(self.nprime):
                            # for p, specially split into two loops to avoid applying code to p "k_arr[q]" times
                            A = (
                                A
                                * self.gen_sample_batch(
                                    t + tau,
                                    T,
                                    x + dw,
                                    mask_tmp,
                                    torch.ones_like(t),
                                    -self.deriv_map[q] - ll - 1,
                                    patch,
                                    self.zeta_map[q],
                                )
                                ** k_arr[q]
                            )
                        for q in range(self.nprime, self.n):
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
        self, patch, coordinate=None, code=None, t=None, discard_outlier=True
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
        for _ in range(batches):
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
                )
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
                yyy.append((yy_tmp.nan_to_num() * mask).sum(dim=1) / mask.sum(dim=1))
            yy.append(torch.stack(yyy, dim=-1))

        return (
            torch.cat(xx, dim=0),
            torch.cat(yy, dim=0),
        )

    def train_and_eval(self, debug_mode=False):
        """
        generate sample and evaluate (plot) NN approximation when debug_mode=True
        """
        for p in range(self.patches):
            # initialize optimizer
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.lr_milestones,
                gamma=self.lr_gamma,
            )

            # loop through epochs
            for epoch in range(self.epochs):
                # clear gradients and evaluate training loss
                optimizer.zero_grad()
                start = time.time()
                # the running statistics for p is very different from time to time, so we do not track them
                self.eval()
                x, y = self.gen_sample(
                    patch=p,
                    coordinate=np.array(range(self.dim)),
                    code=-np.ones((self.dim, self.dim), dtype=int),
                    discard_outlier=True,
                )
                self.train()
                predict = self(x, patch=p)
                x_lo = torch.tensor([self.t_lo] + [self.x_lo] * self.dim, device=self.device)
                x_hi = torch.tensor([self.T] + [self.x_hi] * self.dim, device=self.device)
                states_per_batch = 100000
                unif = (
                    torch.rand((self.dim + 1) * states_per_batch, device=self.device, requires_grad=True)
                         .reshape(states_per_batch, self.dim + 1)
                )
                x = (x_lo + (x_hi - x_lo) * unif).T
                grad = 0
                for (idx, c) in zip(self.deriv_condition_zeta_map, self.deriv_condition_deriv_map):
                    # additional t coordinate
                    grad += self.nth_derivatives(
                        np.insert(c, 0, 0), self(x.T, patch=p)[:, idx], x
                    )
                loss = self.loss(y, predict) + self.loss(grad, torch.zeros_like(grad))

                # additional loss regarding poisson equation
                if self.poisson_loss:
                    poisson_lhs, poisson_rhs = 0, 0
                    order = np.array([0] * (self.dim + 1))
                    for i in range(self.dim):
                        order[i + 1] += 2
                        poisson_lhs += self.nth_derivatives(
                            order, self(x.T, p_or_u="p", patch=p), x
                        )
                        order[i + 1] -= 2
                    for i in range(self.dim):
                        for j in range(self.dim):
                            order[i + 1] += 1
                            tmp = self.nth_derivatives(
                                order, self(x.T, patch=p)[:, j], x
                            )
                            order[i + 1] -= 1
                            order[j + 1] += 1
                            tmp *= self.nth_derivatives(
                                order, self(x.T, patch=p)[:, i], x
                            )
                            order[j + 1] -= 1
                            poisson_rhs -= tmp
                    loss += self.loss(poisson_lhs, poisson_rhs)

                # x, y = self.gen_sample(
                #     patch=p,
                #     coordinate=self.deriv_condition_zeta_map,
                #     code=-self.deriv_condition_deriv_map - 1,
                #     discard_outlier=True,
                # )
                # y = y.sum(dim=-1)
                # loss = self.loss(y, torch.zeros_like(y))

                # update model weights and schedule
                loss.backward()
                optimizer.step()
                scheduler.step()

                self.eval()
                grid = np.linspace(self.x_lo, self.x_hi, 100)
                x_mid = (self.x_lo + self.x_hi) / 2
                t_lo = self.T / 2
                grid_nd = np.concatenate(
                    (
                        t_lo * np.ones((1, 100)),
                        np.expand_dims(grid, axis=0),
                        x_mid * np.ones((self.dim - 1, 100)),
                    ),
                    axis=0,
                ).astype(np.float32)
                nn = (
                    self(
                        torch.tensor(grid_nd.T, device=self.device), patch=0, p_or_u="p"
                    )
                    .detach()
                    .cpu()
                )
                exact = (
                    self.exact_p_fun(torch.tensor(grid_nd.T, device=self.device))
                    .detach()
                    .cpu()
                )
                exact += nn.mean() - exact.mean()
                fig = plt.figure()
                plt.plot(grid, nn, label=f"NN")
                plt.plot(grid, exact, label=f"exact")
                plt.title(f"Epoch {epoch:04}")
                plt.legend(loc="upper left")
                fig.savefig(
                    f"{self.working_dir}/plot/p/epoch_{epoch:04}.png", bbox_inches="tight"
                )
                plt.close()
                nn = (
                    self(
                        torch.tensor(grid_nd.T, device=self.device), patch=0
                    )
                    .detach()
                    .cpu()
                )
                for i in range(self.dim):
                    exact = (
                        self.exact_u_fun(torch.tensor(grid_nd.T, device=self.device), i)
                        .detach()
                        .cpu()
                    )
                    fig = plt.figure()
                    plt.plot(grid, nn[:, i], label=f"NN")
                    plt.plot(grid, exact, label=f"exact")
                    plt.title(f"Epoch {epoch:04}")
                    plt.legend(loc="upper left")
                    fig.savefig(
                        f"{self.working_dir}/plot/u{i}/epoch_{epoch:04}.png", bbox_inches="tight"
                    )
                    plt.close()
                torch.save(
                    self.state_dict(), f"{self.working_dir}/model/epoch_{epoch:04}.pt"
                )

                logging.info(
                    f"Epoch {epoch:3.0f}: one loop takes {time.time() - start:4.0f} seconds with loss {loss.detach():.2E}."
                )


if __name__ == "__main__":
    torch.cuda.empty_cache()  # does this fix the CUDA error??

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
        T, x_lo, x_hi, beta = 0.1, 0, 2 * math.pi, 0.01
        A = B = 0.5
        C = 0.0
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
        lr_gamma=.5,
        branch_nb_path_per_state=100,
        branch_nb_states=100,
        branch_nb_states_per_batch=100,
        beta=beta,
        layers=2,
        batch_normalization=False,
        debug=False,
        branch_activation="tanh",
        poisson_loss=True,
    )
    # model.load_state_dict(torch.load("logs/20220401-154324/model/epoch_2667.pt"))
    model.train_and_eval()
