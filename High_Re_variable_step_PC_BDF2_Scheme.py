import numpy as np
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFiniteElementSpace
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from fealpy.decorator import cartesian, barycentric
from scipy.linalg import *
from scipy.sparse import bmat, spdiags
import random

# --------------------------------------------------
pi = np.pi
C0 = 10
kappa = 0.01
nu = 1e-8  # nu = 1/Re
nx = 5  # space point h = 1/nx
ny = nx
CFL_point = nx*nx  # time points N = 1/h^2
num_range = 1
Time = 1.0
error_order = np.zeros((num_range, 7))
matrix_error = np.zeros((num_range, 4))
matrix_random_tau = np.zeros((num_range, CFL_point))

# --------------------------------------------------
# ---random time steps------------------------------
for k in range(num_range):
    num_points = CFL_point
    sigma = [random.uniform(1/4.8, 1) for _ in range(num_points)]
    tau = sigma / np.sum(sigma)
    matrix_random_tau[k, 0:num_points] = Time * tau


# --------------------------------------------------


class model:
    def __init__(self):
        pass

    def domain(self):
        return np.array([0, 1, 0, 1])

    @cartesian
    def velocity_u(self, p, t):
        """
    	The exact solution
        Parameters
        ---------
        p :
        t :
        Examples
        -------
        p = np.array([0, 1], dtype=np.float64)
        p = np.array([[0, 1], [0.5, 0.5]], dtype=np.float64)
            """
        x = p[..., 0]
        y = p[..., 1]
        val = 2 * y * np.sin(pi * x) ** 2 * (2 * y - 1) * (y - 1) * ((16 * np.cos(4 * t)) / 5 + 24 / 5)
        return val  # velocity solution u

    @cartesian
    def velocity_v(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        val = -y ** 2 * pi * np.sin(2 * pi * x) * (y - 1) ** 2 * ((16 * np.cos(4 * t)) / 5 + 24 / 5)
        return val  # velocity solution v

    @cartesian
    def pressure(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        val = np.cos(pi * y) * np.sin(pi * x) * ((2 * np.cos(4 * t)) / 5 + 3 / 5)
        return val  # pressure solution p

    @cartesian
    def fu(self, p, t):
        """
    	The right hand side of convection-diffusion-reaction equation
        INPUT:
        p: array object,
        t:
        """
        x = p[..., 0]
        y = p[..., 1]
        val = (nu * ((64 * y) / 5 - 32 / 5) * (2 * np.cos(4 * t) + 3) * (
                3 * np.cos(pi * x) ** 2 + y * pi ** 2 * (2 * np.cos(pi * x) ** 2 - 1)
                - y ** 2 * pi ** 2 * (2 * np.cos(pi * x) ** 2 - 1) - 3) + pi * np.cos(pi * x) * np.cos(pi * y) * (
                       (2 * np.cos(4 * t)) / 5 + 3 / 5)
               - (128 * y * np.sin(4 * t) * np.sin(pi * x) ** 2 * (2 * y - 1) * (y - 1)) / 5
               + (512 * y ** 2 * pi * np.cos(pi * x) * np.sin(pi * x) ** 3 * (2 * y - 1) ** 2 * (y - 1) ** 2 * (
                        2 * np.cos(4 * t) + 3) ** 2) / 25
               - 2 * y ** 2 * pi * np.cos(pi * x) * np.sin(pi * x) ** 3 * (y - 1) ** 2 * (2 * np.cos(4 * t) + 3) * (
                       (16 * np.cos(4 * t)) / 5 + 24 / 5) * ((96 * y ** 2) / 5 - (96 * y) / 5 + 16 / 5))
        return val  # the first component of the external force term

    @cartesian
    def fv(self, p, t):
        """
        The right hand side of convection-diffusion-reaction equation
        INPUT:
        p: array object,
        t:
        """
        x = p[..., 0]
        y = p[..., 1]
        val = ((64 * y ** 2 * pi * np.sin(4 * t) * np.sin(2 * pi * x) * (y - 1) ** 2) / 5
               - pi * np.sin(pi * x) * np.sin(pi * y) * ((2 * np.cos(4 * t)) / 5 + 3 / 5)
               - (16 * nu * pi * np.sin(2 * pi * x) * (2 * np.cos(4 * t) + 3) * (
                        2 * pi ** 2 * y ** 4 - 4 * pi ** 2 * y ** 3 + 2 * pi ** 2 * y ** 2 - 6 * y ** 2 + 6 * y - 1)) / 5
               - (256 * y ** 3 * pi ** 2 * np.cos(2 * pi * x) * np.sin(pi * x) ** 2 * (2 * y - 1) * (y - 1) ** 3 * (
                        2 * np.cos(4 * t) + 3) ** 2) / 25
               + (16 * y ** 3 * pi ** 2 * np.sin(2 * pi * x) ** 2 * (y - 1) ** 2 * (2 * np.cos(4 * t) + 3) * (
                        (16 * np.cos(4 * t)) / 5 + 24 / 5) * (2 * y ** 2 - 3 * y + 1)) / 5)
        return val  # the second component of the external force term

    @cartesian
    def dirichlet_u(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        val = 2 * y * np.sin(pi * x) ** 2 * (2 * y - 1) * (y - 1) * ((16 * np.cos(4 * t)) / 5 + 24 / 5)
        return val  # boundary condition

    @cartesian
    def dirichlet_v(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        # val = np.exp(-8 * nu * pi ** 2 * t) * np.cos(2 * pi * x) * np.cos(2 * pi * y)
        val = -y ** 2 * pi * np.sin(2 * pi * x) * (y - 1) ** 2 * ((16 * np.cos(4 * t)) / 5 + 24 / 5)
        return val  # boundary condition

    @cartesian
    def is_dirichlet_boundary(self, p):
        eps = 1e-10
        x = p[..., 0]
        y = p[..., 1]  # to determine whether it is a boundary
        return (np.abs(x) <= eps) | (np.abs(x - 1) <= eps) | (np.abs(y) <= eps) | (np.abs(y - 1) <= eps)


# --------------------------------------------------


# --------------------------------------------------
pde = model()  # create a PDE model object
box = [0, 1.0, 0, 1.0]  # Solution domain
mesh = TriangleMesh.from_box(box, nx, ny)
uspace = LagrangeFiniteElementSpace(mesh, p=2)  # Velocity solution space
pspace = LagrangeFiniteElementSpace(mesh, p=1)  # Pressure solution space
uh_star = uspace.function()
vh_star = uspace.function()
ph_star = pspace.function()
uh_new = uspace.function()
vh_new = uspace.function()
ph_new = pspace.function()
# --------------------------------------------------


# ------numerical quadrature nodes and weights-------
bcs, ws = uspace.integrator.get_quadrature_points_and_weights()
cell_measure = uspace.cellmeasure
# --------------------------------------------------

# ---spatial basis functions of velocity and their degrees of freedom
phi = uspace.basis(bcs)
gphi = uspace.grad_basis(bcs)
cell2dof_u = uspace.cell_to_dof()
gdof_u = uspace.number_of_global_dofs()
# --------------------------------------------------

# ---spatial basis function of pressure and their degrees of freedom
psi = pspace.basis(bcs)
cell2dof_p = pspace.cell_to_dof()
gdof_p = pspace.number_of_global_dofs()
# --------------------------------------------------


# ---set the initial conditions---------------------
@cartesian
def velocity_initial_u(p):
    return pde.velocity_u(p, t=0)


# --------------------------------------------------
@cartesian
def velocity_initial_v(p):
    return pde.velocity_v(p, t=0)


# --------------------------------------------------


# ---(p, div V)-------------------------------------
shape_up = cell2dof_u.shape + cell2dof_p.shape[1:]
# ---V=(\phi, 0)---------------------------------
p_phix = np.einsum('q, qci, qcj, c->cij', ws, gphi[..., 0], psi, cell_measure)
I_p_phix = np.broadcast_to(cell2dof_u[:, :, None], shape=shape_up)
J_p_phix = np.broadcast_to(cell2dof_p[:, None, :], shape=shape_up)
Bu = csr_matrix((p_phix.flat, (I_p_phix.flat, J_p_phix.flat)), shape=(gdof_u, gdof_p))
# ---V=(0, \phi)---------------------------------
p_phiy = np.einsum('q, qci, qcj, c->cij', ws, gphi[..., 1], psi, cell_measure)
I_p_phiy = np.broadcast_to(cell2dof_u[:, :, None], shape=shape_up)
J_p_phiy = np.broadcast_to(cell2dof_p[:, None, :], shape=shape_up)
Bv = csr_matrix((p_phiy.flat, (I_p_phiy.flat, J_p_phiy.flat)), shape=(gdof_u, gdof_p))
# --------------------------------------------------

# ---(phi,1)=0--->B0*p=0----------------------------
b_p0 = np.einsum('i, ijk, j -> jk', ws, psi, cell_measure)
B_p0 = np.zeros(gdof_p, dtype=np.float64)
np.add.at(B_p0, cell2dof_p, b_p0)
# --------------------------------------------------


# ---stabilization \kappa(\nabla\cdot U,\nabla\cdot V)
# ---(\phi_x,\phi_x)------------------------
shape_uu = cell2dof_u.shape + cell2dof_u.shape[1:]
phix_x = np.einsum('q, qci, qcj, c->cij', ws, gphi[..., 0], gphi[..., 0], cell_measure)
I_phix_x = np.broadcast_to(cell2dof_u[:, :, None], shape=shape_uu)
J_phix_x = np.broadcast_to(cell2dof_u[:, None, :], shape=shape_uu)
K1 = csr_matrix((phix_x.flat, (I_phix_x.flat, J_phix_x.flat)), shape=(gdof_u, gdof_u))
Dx = K1
K1 = kappa * K1
# --------------------------------------------------

# ---(\phi_x,\phi_y)-------------------------
phix_y = np.einsum('q, qci, qcj, c->cij', ws, gphi[..., 0], gphi[..., 1], cell_measure)
I_phix_y = np.broadcast_to(cell2dof_u[:, :, None], shape=shape_uu)
J_phix_y = np.broadcast_to(cell2dof_u[:, None, :], shape=shape_uu)
K2 = csr_matrix((phix_y.flat, (I_phix_y.flat, J_phix_y.flat)), shape=(gdof_u, gdof_u))
Dy = K2
K2 = kappa * K2
K2T = K2.T
# --------------------------------------------------

# # ---(\phi_y,\phi_y)-------------------------
phiy_y = np.einsum('q, qci, qcj, c->cij', ws, gphi[..., 1], gphi[..., 1], cell_measure)
I_phiy_y = np.broadcast_to(cell2dof_u[:, :, None], shape=shape_uu)
J_phiy_y = np.broadcast_to(cell2dof_u[:, None, :], shape=shape_uu)
K3 = csr_matrix((phiy_y.flat, (I_phiy_y.flat, J_phiy_y.flat)), shape=(gdof_u, gdof_u))
K3 = kappa * K3
# --------------------------------------------------


# --------------------------------------------------
def nonlinear_skew_term(uh0, vh0):
    val_u = uh0(bcs)
    val_v = vh0(bcs)
    gval_u = uh0.grad_value(bcs)
    gval_v = vh0.grad_value(bcs)
    # ---(U\nabla U, V)--------------------------
    conv_u = val_u * gval_u[..., 0] + val_v * gval_u[..., 1]
    conv_v = val_u * gval_v[..., 0] + val_v * gval_v[..., 1]
    b_conv_u = np.einsum('i, ij, ijk, j -> jk', ws, conv_u, phi, cell_measure)
    B_conv_u = np.zeros(gdof_u, dtype=np.float64)
    np.add.at(B_conv_u, cell2dof_u, b_conv_u)
    b_conv_v = np.einsum('i, ij, ijk, j -> jk', ws, conv_v, phi, cell_measure)
    B_conv_v = np.zeros(gdof_u, dtype=np.float64)
    np.add.at(B_conv_v, cell2dof_u, b_conv_v)
    # ---((div U) U, V)--------------------------
    val_div = gval_u[..., 0] + gval_v[..., 1]
    skew_u = val_div * val_u
    skew_v = val_div * val_v
    b_skew_u = np.einsum('i, ij, ijk, j -> jk', ws, skew_u, phi, cell_measure)
    b_skew_v = np.einsum('i, ij, ijk, j -> jk', ws, skew_v, phi, cell_measure)
    B_skew_u = np.zeros(gdof_u, dtype=np.float64)
    np.add.at(B_skew_u, cell2dof_u, b_skew_u)
    B_skew_v = np.zeros(gdof_u, dtype=np.float64)
    np.add.at(B_skew_v, cell2dof_u, b_skew_v)
    # -----------------------------------------------
    Cu = B_conv_u + 0.5 * B_skew_u
    Cv = B_conv_v + 0.5 * B_skew_v
    return Cu, Cv
    # --------------------------------------------------


# ---Compute the sav variable---------------------------
def val_l2_norm(uh0, vh0):
    val_u = uh0(bcs)
    val_v = vh0(bcs)
    val_L2 = C0 + 0.5 * np.einsum('i, ij, j', ws, val_u ** 2 + val_v ** 2, cell_measure)
    return val_L2
    # --------------------------------------------------


# ---compute ||\nabla U|| variable----------------------
def val_h1_norm(uh0, vh0):
    gval_u = uh0.grad_value(bcs)
    gval_v = vh0.grad_value(bcs)
    gval_ux = gval_u[..., 0]
    gval_uy = gval_u[..., 1]
    gval_vx = gval_v[..., 0]
    gval_vy = gval_v[..., 1]
    gval_L2 = np.einsum('i, ij, j', ws, gval_ux ** 2 + gval_uy ** 2 + gval_vx ** 2 + gval_vy ** 2, cell_measure)
    return gval_L2
    # --------------------------------------------------

# ---transform barycentric coordinates to physical coordinates
pp = uspace.mesh.bc_to_point(bcs)

# ---compute(fu,V) and (fv,V)---------------------------
def val_force_product(uh0, vh0, fu0, fv0):
    val_u = uh0(bcs)
    val_v = vh0(bcs)
    val_fu = fu0(pp)
    val_fv = fv0(pp)
    force_L2 = np.einsum('i, ij, j', ws, val_fu * val_u + val_fv * val_v, cell_measure)
    return force_L2

# --------------------------------------------------
M = uspace.mass_matrix()   # assemble the mass matrix
S = uspace.stiff_matrix()  # assemble the stiff matrix
S_nu = nu * S
# --------------------------------------------------
gdof = gdof_p + 2 * gdof_u
SK1_nu = nu * S + K1
SK3_nu = nu * S + K3
# ---handle the boundary conditions-----------------
u_isBdDof = uspace.is_boundary_dof()
v_isBdDof = u_isBdDof
p_isBdDof = np.zeros(gdof_p + 1, dtype=bool)
isBdDof = np.hstack([u_isBdDof, v_isBdDof, p_isBdDof])
ipoint = uspace.interpolation_points()
bd_index = np.zeros(gdof + 1, dtype=np.int_)
bd_index[isBdDof] = 1
# ---mark as 1 if it is a boundary point; otherwise, mark as 0.
Tbd = spdiags(bd_index, 0, gdof + 1, gdof + 1)
T0 = spdiags(1 - bd_index, 0, gdof + 1, gdof + 1)
# ---(phi,1)=0----------------------------
bp0 = np.einsum('i, ijk, j -> jk', ws, psi, cell_measure)
Bp0 = np.zeros(gdof_p, dtype=np.float64)
np.add.at(Bp0, cell2dof_p, bp0)
# --------------------------------------------------
zero_as_vector = np.array([0])
Bp0_column = np.append(Bp0, zero_as_vector)
P0 = np.zeros((gdof_p, gdof_p), dtype=np.float64)
P0 = np.vstack((P0, Bp0))
P0 = np.hstack((P0, Bp0_column.reshape(-1, 1)))
P0 = csr_matrix(P0)
# --------------------------------------------------
new_Bu_shape = (Bu.shape[0], Bu.shape[1] + 1)
Bu = csr_matrix((Bu.data, Bu.indices, Bu.indptr), shape=new_Bu_shape)
new_Bv_shape = (Bv.shape[0], Bv.shape[1] + 1)
Bv = csr_matrix((Bv.data, Bv.indices, Bv.indptr), shape=new_Bv_shape)
# --------------------------------------------------

# --------------------------------------------------
for k in range(num_range):
    print("range = ", k)
    # ---temporal mesh partition---------------------
    num_points = CFL_point
    random_tau = matrix_random_tau[k, 0:num_points]
    tau = random_tau[0]
    time_grid = 0
    tn_old = 0  # ---old step
    tn_new = 0  # ---new step
    tau_old = tau
    tau_new = tau
    tau_max = np.max(random_tau)
    L2Error_p = 0  # ---initialize the accumulated pressure error
    L2Error_U = 0
    L2Error_P = 0
    L1Error_sav = 0
    counter = 1  # ---indicate whether to use BDF1 or BDF2
    uh_initial_data = uspace.interpolation(velocity_initial_u)
    vh_initial_data = uspace.interpolation(velocity_initial_v)
    uh_old = uspace.function()
    vh_old = uspace.function()
    uh_old1 = uspace.function()
    vh_old1 = uspace.function()
    uh_hat = uspace.function()
    vh_hat = uspace.function()
    En = val_l2_norm(uh_initial_data, vh_initial_data)
    gamma = np.array([1])
    sav = En
    sav_old = En
    # --------------------------------------------------
    for n in range(0, num_points):
        tau_new = random_tau[n]
        tn_new = tn_old + tau_new
        print("time = ", tn_new)


        # --------------------------------------------------

        @cartesian
        def dirichlet_u(p):
            return pde.dirichlet_u(p, tn_new)


        @cartesian
        def dirichlet_v(p):
            return pde.dirichlet_v(p, tn_new)


        # --------------------------------------------------
        @cartesian
        def source_u(p):
            return pde.fu(p, tn_new)


        @cartesian
        def source_v(p):
            return pde.fv(p, tn_new)


        # --------------------------------------------------
        if counter == 1:  # BDF1 for n = 1
            uh_old[:] = uh_initial_data
            vh_old[:] = vh_initial_data
            Cu_old, Cv_old = nonlinear_skew_term(uh_old, vh_old)
            M_BDF1 = 1 / tau_new * M
            Fu = M_BDF1 @ uh_old - Cu_old + uspace.source_vector(source_u)
            Fv = M_BDF1 @ vh_old - Cv_old + uspace.source_vector(source_v)
            A = bmat([[M_BDF1 + SK1_nu, K2, -Bu],
                      [K2T, M_BDF1 + SK3_nu, -Bv],
                      [-Bu.T, -Bv.T, P0]], format='csr')
        else:  # BDF2 for n > 1
            rn = tau_new / tau_old
            bn0 = (1 + 2 * rn) / (1 + rn) / tau_new
            bn1 = -rn ** 2 / (1 + rn) / tau_new
            bn2 = bn0 - bn1
            uh_hat[:] = (1 + rn) * uh_old - rn * uh_old1
            vh_hat[:] = (1 + rn) * vh_old - rn * vh_old1
            Cu_old, Cv_old = nonlinear_skew_term(uh_hat, vh_hat)
            Fu = M @ (bn2 * uh_old + bn1 * uh_old1) - Cu_old + uspace.source_vector(source_u)  # 组装右端项
            Fv = M @ (bn2 * vh_old + bn1 * vh_old1) - Cv_old + uspace.source_vector(source_v)
            M_BDF2 = bn0 * M
            A = bmat([[M_BDF2 + SK1_nu, K2, -Bu],
                      [K2T, M_BDF2 + SK3_nu, -Bv],
                      [-Bu.T, -Bv.T, P0]], format='csr')
        # --------------------------------------------------

        # --------------------------------------------------
        x_uvp = np.zeros(gdof + 1, np.float64)
        x0_uvp = np.zeros(gdof + 1, np.float64)
        u_ipoint = dirichlet_u(ipoint)
        v_ipoint = dirichlet_v(ipoint)
        x_uvp[0:gdof_u][u_isBdDof] = u_ipoint[:][u_isBdDof]
        x_uvp[gdof_u:2 * gdof_u][v_isBdDof] = v_ipoint[:][v_isBdDof]
        # --------------------------------------------------
        FF = np.r_['0', Fu, Fv, np.zeros(gdof_p + 1)]
        FF = FF - A @ x_uvp
        A0 = T0 @ A @ T0 + Tbd
        FF[isBdDof] = x_uvp[isBdDof]
        x0_uvp[:] = spsolve(A0, FF)
        uh_star[:] = x0_uvp[:gdof_u]
        vh_star[:] = x0_uvp[gdof_u:2 * gdof_u]
        ph_star[:] = x0_uvp[2 * gdof_u:-1]
        # ---compute sav variable------------------------------------
        val_L2_star = val_l2_norm(uh_star, vh_star)
        val_H1_star = val_h1_norm(uh_star, vh_star)
        force_L2_star = val_force_product(uh_star, vh_star, source_u, source_v)
        sav_star = sav_old / (1 + tau_new * (nu * val_H1_star - force_L2_star) / val_L2_star)
        gamma_new = sav_star / val_L2_star
        Gamma_new = 1 - (1 - gamma_new) ** 2
        # -------------------------------------------------
        uh_new[:] = Gamma_new * uh_star
        vh_new[:] = Gamma_new * vh_star
        # ---compute relaxation sav------------------------
        dissipation_law = (sav_star - val_L2_star)/tau_new + gamma_new * nu * val_H1_star
        if sav_star >= val_L2_star:
            weight_new = 0
            sav_new = val_L2_star
        elif sav_star < val_L2_star and dissipation_law >= 0:
            weight_new = 0
            sav_new = val_L2_star
        else:
            weight_new = 1 - tau_new * gamma_new * nu * val_H1_star / (val_L2_star - sav_star)
            sav_new = weight_new * sav_star + (1 - weight_new) * val_L2_star
        # # -------------------------------------------------
        if counter == 1:
            Cu_new, Cv_new = nonlinear_skew_term(uh_new, vh_new)
            M_BDF1 = 1 / tau_new * M
            Fu = M_BDF1 @ uh_old - Cu_new + uspace.source_vector(source_u)
            Fv = M_BDF1 @ vh_old - Cv_new + uspace.source_vector(source_v)
            Ap = bmat([[M_BDF1 + SK1_nu, K2, -Bu],
                       [K2T, M_BDF1 + SK3_nu, -Bv],
                       [-Bu.T, -Bv.T, P0]], format='csr')
        else:
            rn = tau_new / tau_old
            bn0 = (1 + 2 * rn) / (1 + rn) / tau_new
            bn1 = -rn ** 2 / (1 + rn) / tau_new
            bn2 = bn0 - bn1
            Cu_new, Cv_new = nonlinear_skew_term(uh_new, vh_new)
            Fu = M @ (bn2 * uh_old + bn1 * uh_old1) - Cu_new + uspace.source_vector(source_u)  # 组装右端项
            Fv = M @ (bn2 * vh_old + bn1 * vh_old1) - Cv_new + uspace.source_vector(source_v)
            M_BDF2 = bn0 * M
            Ap = bmat([[M_BDF2 + SK1_nu, K2, -Bu],
                       [K2T, M_BDF2 + SK3_nu, -Bv],
                       [-Bu.T, -Bv.T, P0]], format='csr')
        # -------------------------------------------------
        x_uvp = np.zeros(gdof + 1, np.float64)
        x0_uvp = np.zeros(gdof + 1, np.float64)
        u_ipoint = dirichlet_u(ipoint)
        v_ipoint = dirichlet_v(ipoint)
        x_uvp[0:gdof_u][u_isBdDof] = u_ipoint[:][u_isBdDof]
        x_uvp[gdof_u:2 * gdof_u][v_isBdDof] = v_ipoint[:][v_isBdDof]
        # --------------------------------------------------
        FF = np.r_['0', Fu, Fv, np.zeros(gdof_p + 1)]
        FF = FF - Ap @ x_uvp
        A0 = T0 @ Ap @ T0 + Tbd
        FF[isBdDof] = x_uvp[isBdDof]
        x0_uvp[:] = spsolve(A0, FF)
        ph_new[:] = x0_uvp[2 * gdof_u:-1]
        # -------------------------------------------------
        uh_old1[:] = uh_old
        vh_old1[:] = vh_old
        uh_old[:] = uh_new
        vh_old[:] = vh_new
        sav_old = sav_new
        tn_old = tn_new
        tau_old = tau_new
        gamma = np.append(gamma, gamma_new)
        sav = np.append(sav, sav_new)
        time_grid = np.append(time_grid, tn_new)
        En = np.append(En, Gamma_new ** 2 * (val_L2_star - C0) + C0)
        counter = counter + 1
        # --------------------------------------------------


        # --------------------------------------------------
        # --------------------------------------------------
        @cartesian
        def pressure_exact(p):
            return pde.pressure(p, t=tn_new)


        # --------------------------------------------------
        @cartesian
        def velocity_u_exact(p):
            return pde.velocity_u(p, t=tn_new)


        # --------------------------------------------------
        @cartesian
        def velocity_v_exact(p):
            return pde.velocity_v(p, t=tn_new)


        # --------------------------------------------------
        L2Error = pspace.integralalg.error(pressure_exact, ph_new)
        L2Error_p = np.append(L2Error_p, L2Error)
        L2Error_P = L2Error_P + tau_new * (pspace.integralalg.error(pressure_exact, ph_new)) ** 2
        L2Error_u = uspace.integralalg.error(velocity_u_exact, uh_new)
        L2Error_v = uspace.integralalg.error(velocity_v_exact, vh_new)
        L2Error_U = np.append(L2Error_U, np.sqrt(L2Error_u ** 2 + L2Error_v ** 2))
        L1Error_sav = np.append(L1Error_sav, np.abs(1 - gamma_new))
    # --------------------------------------------------
    matrix_error[k, 0] = 1/nx
    matrix_error[k, 1] = L2Error_U[-1]
    matrix_error[k, 2] = np.sqrt(L2Error_P)
    matrix_error[k, 3] = np.max(L1Error_sav)
    print("matrix_error = ", matrix_error)
    # --------------------------------------------------
fmt = ['%.2f', '%.2e', '%.2e', '%.2e']
header = "h Error_u  Error_p  Error_sav"
with open('PC-BDF2_output.txt', 'w') as file:
    for row in matrix_error:
        file.write(" ".join(f"{item:.2f}" for item in row) + "\n")
np.savetxt('PC-BDF2_output.txt', matrix_error, header=header, fmt=fmt, delimiter='\t', comments='')
