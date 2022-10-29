# Deep branching solver for PDE system
Authors: Jiang Yu Nguwi and Nicolas Privault.

If this code is used for research purposes, please cite as \
J.Y. Nguwi, G. Penent, and N. Privault.
Numerical solution of the incompressible Navier-Stokes equation by deep branching.
<br/><br/>

Deep branching solver based on [[NPP22]](#nguwi2022deepbranching)
aims to solve system of fully nonlinear PDE of the form\
<img src="https://latex.codecogs.com/svg.image?0&space;=&space;\partial_t&space;u_i(t,x)&space;&plus;&space;\nu&space;\Delta&space;u_i(t,x)&space;&space;&space;&space;&plus;&space;f_i\big(&space;&space;&space;&space;\partial_{\alpha^1}u_0&space;(t,x)&space;&space;&space;&space;,&space;&space;&space;&space;\ldots&space;,&space;&space;&space;&space;\partial_{\alpha^q&space;}u_0&space;(t,x)&space;&space;&space;&space;,&space;&space;&space;&space;&space;&space;&space;&space;\partial_{\alpha^{q&plus;1}}u_{\beta^{q&plus;1}}(t,x)&space;&space;&space;&space;&space;&space;&space;&space;,&space;&space;&space;&space;&space;&space;&space;&space;\ldots&space;,&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;&space;\partial_{\alpha^n}u_{\beta^n}(t,x)&space;\big)," />\
with\
<img src="https://latex.codecogs.com/svg.image?&space;&space;\Delta&space;u_0(t,&space;x)&space;&space;=&space;-\sum\limits_{i=1}^d&space;\partial_{1_i}&space;u_j(t,&space;x)&space;\partial_{1_j}&space;u_i(t,&space;x)," />\
and\
<img src="https://latex.codecogs.com/svg.image?u_i(T,x)&space;=&space;g_i&space;(x),\quad&space;(t,x)&space;=&space;(t,x_1,&space;\ldots,&space;x_d)&space;\in&space;[0,T]&space;\times&space;\mathbb{R}^d,\quad&space;i&space;=&space;1,\ldots&space;,&space;d." />

## Quick Start
Click one of the following links for quick start.

* <a href="https://colab.research.google.com/github/nguwijy/deep_navier_stokes/blob/main/main.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

* <a href="https://mybinder.org/v2/gh/nguwijy/deep_navier_stokes/main?labpath=notebooks%2Findex.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Launch binder" /></a>

* <a href="https://nbviewer.org/github/nguwijy/deep_navier_stokes/blob/main/main.ipynb"><img src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg" alt="Render nbviewer" /></a>

* [github.com's notebook viewer](https://github.com/nguwijy/deep_navier_stokes/blob/main/main.ipynb) also works but it's slower and may fail to open large notebooks.


## References
<a id="nguwi2022deepbranching">[NPP22]</a>
J.Y. Nguwi, G. Penent, and N. Privault.
Numerical solution of the incompressible Navier-Stokes equation by deep branching.
