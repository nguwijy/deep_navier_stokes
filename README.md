# Deep branching solver for PDE system
Deep branching solver based on [[NPP22]](#nguwi2022deepbranching)
aims to solve system of fully nonlinear PDE of the form\
<img src="https://latex.codecogs.com/svg.image?0&space;=&space;\partial_t&space;u_i(t,x)&space;&plus;&space;\nu&space;\Delta&space;u_i(t,x)&space;&space;&space;&space;&plus;&space;f_i\big(&space;&space;&space;&space;\partial_{\alpha^1}u_0&space;(t,x)&space;&space;&space;&space;,&space;&space;&space;&space;\ldots&space;,&space;&space;&space;&space;\partial_{\alpha^q&space;}u_0&space;(t,x)&space;&space;&space;&space;,&space;&space;&space;&space;&space;&space;&space;&space;\partial_{\alpha^{q&plus;1}}u_{\beta^{q&plus;1}}(t,x)&space;&space;&space;&space;&space;&space;&space;&space;,&space;&space;&space;&space;&space;&space;&space;&space;\ldots&space;,&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;\partial_{\alpha^n}u_{\beta^n}(t,x)&space;\big)," />\
with\
<img src="https://latex.codecogs.com/svg.image?&space;&space;\Delta&space;u_0(t,&space;x)&space;&space;=&space;-\sum\limits_{i=1}^d&space;\partial_{1_i}&space;u_j(t,&space;x)&space;\partial_{1_j}&space;u_i(t,&space;x)," />\
and\
<img src="https://latex.codecogs.com/svg.image?u_i(T,x)&space;=&space;g_i&space;(x),\quad&space;(t,x)&space;=&space;(t,x_1,&space;\ldots,&space;x_d)&space;\in&space;[0,T]&space;\times&space;\mathbb{R}^d,\quad&space;i&space;=&space;1,\ldots&space;,&space;d." />

We let
<img src="http://latex.codecogs.com/svg.latex?d&space;=&space;2," />
<img src="http://latex.codecogs.com/svg.latex?T&space;=&space;.25," />
<img src="http://latex.codecogs.com/svg.latex?\nu&space;=&space;1," />
<img src="https://latex.codecogs.com/svg.image?\alpha_1&space;=&space;1_1," />
<img src="https://latex.codecogs.com/svg.image?\alpha_2&space;=&space;1_2," />
<img src="https://latex.codecogs.com/svg.image?\alpha_3&space;=&space;0," />
<img src="https://latex.codecogs.com/svg.image?\alpha_4&space;=&space;0," />
<img src="https://latex.codecogs.com/svg.image?\alpha_5&space;=&space;1_1," />
<img src="https://latex.codecogs.com/svg.image?\alpha_6&space;=&space;1_2," />
<img src="https://latex.codecogs.com/svg.image?\alpha_7&space;=&space;1_1," />
<img src="https://latex.codecogs.com/svg.image?\alpha_8&space;=&space;1_2," />
<img src="https://latex.codecogs.com/svg.image?\beta_3&space;=&space;1," />
<img src="https://latex.codecogs.com/svg.image?\beta_4&space;=&space;2,"/>
<img src="https://latex.codecogs.com/svg.image?\beta_5&space;=&space;1," />
<img src="https://latex.codecogs.com/svg.image?\beta_6&space;=&space;1," />
<img src="https://latex.codecogs.com/svg.image?\beta_7&space;=&space;2," />
<img src="https://latex.codecogs.com/svg.image?\beta_8&space;=&space;2," />
<img src="https://latex.codecogs.com/svg.image?&space;&space;f_i\big(&space;x_1,&space;x_2,&space;y_1,&space;y_2,&space;z^{(1)}_1,&space;z^{(2)}_1,&space;z^{(1)}_2,&space;z^{(2)}_2&space;\big)&space;=&space;-&space;x_i&space;-&space;\sum\limits_{j=1}^2&space;y_j&space;z_i^{(j)}," />\
so that the PDE system becomes the 2D incompresible Navier-Stokes equation.

In this case, we additionally have the divergence free constraint given by\
<img src="https://latex.codecogs.com/svg.image?&space;&space;\sum\limits_{i=1}^d&space;\partial_{1_i}&space;u_j(t,&space;x)&space;\partial_{1_j}&space;u_i(t,&space;x)=0." />

We further assume the Taylor-Green terminal condition
<img src="https://latex.codecogs.com/svg.image?g_1(x_1,&space;x_2)&space;=&space;-\cos(x_1)&space;\sin(x_2)" />
and
<img src="https://latex.codecogs.com/svg.image?g_2(x_1,&space;x_2)&space;=&space;\sin(x_1)&space;\cos(x_2)" />
so that the PDE system admits the true solution of\
<img src="https://latex.codecogs.com/svg.image?u_1(t,&space;x_1,&space;x_2)&space;=&space;-\cos(x_1)&space;\sin(x_2)&space;\exp(-2&space;\nu&space;(T&space;-&space;t))" />
and
<img src="https://latex.codecogs.com/svg.image?u_2(t,&space;x_1,&space;x_2)&space;=&space;\sin(x_1)&space;\cos(x_2)&space;\exp(-2&space;\nu&space;(T&space;-&space;t))." />

For illustration purposes,
suppose that we are only interested in the solution u(T/2, x) for
<img src="http://latex.codecogs.com/svg.latex?x&space;\in&space;[0,&space;2\pi]&space;\times&space;\{&space;\pi\}." />

## Using deep branching solver
There are two ways to utilize the deep branching solver:
1. Edit the templates inside the `__main__` environment
   in `branch.py`, then run `python branch.py` from your terminal.
2. Write your own code and import the solver to your code via `from branch import Net`,
   see the sections [defining the derivatives map and the functions](#defining-the-derivatives-map-and-the-functions)
   and [training the model](#training-the-model).

## Defining the derivatives map and the functions
Functions f and g must be written in the PyTorch framework, e.g.
```python
import math
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T, x_lo, x_hi, nu = 0.25, 0, 2 * math.pi, 1.0
dim = 2
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


def f_fun(y, i):
    f = -y[i]
    for j in range(dim):
        f += -y[dim + j] * y[2 * dim + dim * i + j]
    return f

def g_fun(x, i):
    if i == 0:
        return -torch.cos(x[0]) * torch.sin(x[1])
    else:
        return torch.sin(x[0]) * torch.cos(x[1])
```

## Training the model
Next, we are ready to initialize the model and to train it.
After the training,
the graph comparing deep branching solution and the true solution
is plotted.
```python
from branch import Net

# the exact functions are only used for plotting
def exact_u_fun(x, i):
     if i == 0:
         return -torch.cos(x[:, 1]) * torch.sin(x[:, 2]) * torch.exp(-2 * nu * (T - x[:, 0]))
     else:
         return torch.sin(x[:, 1]) * torch.cos(x[:, 2]) * torch.exp(-2 * nu * (T - x[:, 0]))

def exact_p_fun(x):
     return (
         -1
         / 4
         * torch.exp(-4 * nu * (T - x[:, 0]))
         * (torch.cos(2 * x[:, 1]) + torch.cos(2 * x[:, 2]))
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
    nu=nu,
    plot_y_lim=[[-1, 1], [-1, 1]],
)
model.train_and_eval()
```
The resulting plot is available at /path/to/logs/plot/:\
![image](logs/20220421-231219/plot/u0/epoch_9999.png)
![image](logs/20220421-231219/plot/u1/epoch_9999.png)

## Loading trained model
Suppose we would like to load a trained model,
either to reuse it or to study the neural function.
We can do so using the `Net.load_dict()` function, i.e.
```python
model = Net(...)
model.load_dict(/path/to/model/)
```

## References
<a id="nguwi2022deepbranching">[NPP22]</a>
J.Y. Nguwi, G. Penent, and N. Privault.
Numerical solution of the incompressible Navier-Stokes equation by deep branching.
