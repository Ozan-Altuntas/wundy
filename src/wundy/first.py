from typing import Any

import numpy as np
from numpy.typing import NDArray

from .schemas import DIRICHLET
from .schemas import NEUMANN

# ---------------------------------------------------------------------------
# Material response (linear elastic bar)
# ---------------------------------------------------------------------------


def material_tangent_stiffness_elastic(material: dict[str, Any]) -> float:
    """Return the 1D elastic tangent stiffness (Young's modulus).

    Parameters
    ----------
    material:
        Material dictionary with a ``"parameters"`` sub-dictionary containing
        the key ``"E"`` for Young's modulus.

    Returns
    -------
    float
        Young's modulus :math:`E`.
    """
    return float(material["parameters"]["E"])


def material_stress_elastic(material: dict[str, Any], strain: float) -> float:
    """Compute Cauchy stress for a 1D linear elastic material.

    Parameters
    ----------
    material:
        Material dictionary compatible with
        :func:`material_tangent_stiffness_elastic`.
    strain:
        Axial strain :math:`\\varepsilon`.

    Returns
    -------
    float
        Axial stress :math:`\\sigma = E \\varepsilon`.
    """
    E = material_tangent_stiffness_elastic(material)
    return E * strain


# ---------------------------------------------------------------------------
# Element response for T1D1 (two-node 1D bar) with Gauss quadrature
# ---------------------------------------------------------------------------


def gauss_points_1d_two_point() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return abscissae and weights for 2-point Gauss quadrature on [-1, 1].

    This rule is exact for polynomials up to degree three.  For the 2-node
    bar element used here the strain-displacement matrix is constant, so the
    integrands for stiffness and internal force are at most quadratic and
    are therefore integrated exactly by this rule.
    """
    xi = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)], dtype=float)
    w = np.ones(2, dtype=float)
    return xi, w


def element_stiffness_t1d1(
    xe: NDArray[np.float64],
    area: float,
    E: float,
) -> NDArray[np.float64]:
    """Compute the 2×2 stiffness matrix for a T1D1 bar element with Gauss quadrature.

    The element is treated as a 2-node isoparametric bar on the reference
    interval :math:`\\xi \\in [-1, 1]`.  The stiffness matrix is

    .. math::

        k_e = \\int_{-1}^{1} B(\\xi)^T\\, E A\\, B(\\xi)\\, J\\,\\mathrm{d}\\xi ,

    where :math:`B` is the strain–displacement matrix and :math:`J` is the
    Jacobian (here :math:`J = h_e / 2`, the half element length).  Because
    :math:`B` is constant for the linear T1D1 element, the 2-point Gauss
    rule integrates this expression exactly.

    Parameters
    ----------
    xe:
        Coordinates of the two element nodes with shape ``(2, num_dim)``.
        Only the first column ``xe[:, 0]`` (the x-coordinate) is used.
    area:
        Cross-sectional area :math:`A` of the bar.
    E:
        Young’s modulus :math:`E`.

    Returns
    -------
    ndarray
        Element stiffness matrix :math:`k_e` with shape ``(2, 2)``.
    """
    he = float(xe[1, 0] - xe[0, 0])
    if np.isclose(he, 0.0):
        raise ValueError("Zero-length element detected in element_stiffness_t1d1")

    # 1D linear shape functions: N1 = (1-ξ)/2, N2 = (1+ξ)/2
    # Their derivatives with respect to ξ are constant.
    dN_dxi = np.array([-0.5, 0.5], dtype=float)

    xi_gp, w_gp = gauss_points_1d_two_point()
    J = he / 2.0  # dx/dξ for the linear mapping

    ke = np.zeros((2, 2), dtype=float)
    for a in range(xi_gp.size):
        # B = dN/dx = dN/dξ * dξ/dx = dN/dξ / J
        B = dN_dxi / J  # shape (2,)
        ke += (B[:, None] @ B[None, :]) * (E * area * J * w_gp[a])

    return ke


def element_internal_force_t1d1(
    xe: NDArray[np.float64],
    area: float,
    material: dict[str, Any],
    ue: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute internal nodal forces for a T1D1 element using Gauss quadrature.

    The internal nodal force vector is obtained from

    .. math::

        f_\\text{int} = \\int_{-1}^{1} B(\\xi)^T \\, \\sigma(\\xi)\\, A\\, J\\,\\mathrm{d}\\xi ,

    where :math:`\\sigma(\\xi) = E\\,\\varepsilon(\\xi)` and
    :math:`\\varepsilon(\\xi) = B(\\xi) u_e`.  For the linear bar element the
    strain and stress are constant, but we still perform the integral using
    the same 2-point Gauss rule as in :func:`element_stiffness_t1d1` to
    mirror the standard finite element formulation.

    Parameters
    ----------
    xe:
        Coordinates of the two element nodes with shape ``(2, num_dim)``.
    area:
        Cross-sectional area :math:`A` of the bar.
    material:
        Material dictionary compatible with
        :func:`material_tangent_stiffness_elastic`.
    ue:
        Element displacement vector with shape ``(2,)``.

    Returns
    -------
    ndarray
        Internal nodal force vector :math:`f_\\text{int}` with shape ``(2,)``.
    """
    he = float(xe[1, 0] - xe[0, 0])
    if np.isclose(he, 0.0):
        raise ValueError("Zero-length element detected in element_internal_force_t1d1")

    dN_dxi = np.array([-0.5, 0.5], dtype=float)
    xi_gp, w_gp = gauss_points_1d_two_point()
    J = he / 2.0

    fint = np.zeros(2, dtype=float)
    for a in range(xi_gp.size):
        B = dN_dxi / J
        strain = float(B @ ue)
        stress = material_stress_elastic(material, strain)
        # f_int contribution: B^T * σ * A * J * w
        fint += B * stress * area * J * w_gp[a]

    return fint


def element_external_force_t1d1(q: float, he: float) -> NDArray[np.float64]:
    """Compute equivalent nodal forces for a uniform line load.

    Parameters
    ----------
    q:
        Uniform line load :math:`q` (force per unit length) in the axial
        direction.
    he:
        Element length :math:`h_e`.

    Returns
    -------
    ndarray
        Two-component nodal force vector for the element.
    """
    return q * he / 2.0 * np.ones(2, dtype=float)


# ---------------------------------------------------------------------------
# Global assembly helpers
# ---------------------------------------------------------------------------


def apply_neumann_bcs(
    F: NDArray[np.float64],
    bcs: list[dict[str, Any]],
    dof_per_node: int,
) -> None:
    """Add Neumann (force) boundary conditions to the global force vector.

    For each boundary–condition entry of type :data:`NEUMANN`, this routine
    loops over the listed nodes, computes the corresponding global degree of
    freedom, and adds the prescribed value to the force vector ``F`` in place.
    """
    for bc in bcs:
        if bc["type"] == NEUMANN:
            for n in bc["nodes"]:
                I = global_dof(n, bc["local_dof"], dof_per_node)
                F[I] += float(bc["value"])


def apply_distributed_loads(
    F: NDArray[np.float64],
    dloads: list[dict[str, Any]],
    coords: NDArray[np.float64],
    blocks: list[dict[str, Any]],
    block_elem_map: dict[int, tuple[int, int]],
    materials: dict[str, Any],
    dof_per_node: int,
) -> None:
    """Assemble equivalent nodal forces for all distributed loads.

    The input ``dloads`` is a list of preprocessed distributed–load
    dictionaries.  For each load and each affected element this routine

    * finds the element and its nodes via ``block_elem_map`` and ``blocks``,
    * computes the element length and area,
    * determines the scalar line load ``q`` based on the load ``type``
      (``"BX"`` or ``"GRAV"``) and the requested direction, and
    * uses :func:`element_external_force_t1d1` to obtain equivalent nodal
      forces which are then added to ``F`` in place.
    """
    for dload in dloads:
        dtype = dload["type"]
        direction = np.array(dload["direction"], dtype=float)
        if direction.size != 1:
            raise ValueError(f"1D problem expects one direction component, got {direction}")
        sign = float(np.sign(direction[0]))
        if sign == 0.0:
            raise ValueError(f"dload direction must be ±1, got {direction[0]}")

        for eid in dload["elements"]:
            if eid not in block_elem_map:
                raise ValueError(
                    f"Element {eid} in distributed load "
                    f"{dload['name']} not found in any element block"
                )

            block_index, local_index = block_elem_map[eid]
            block = blocks[block_index]
            nodes = block["connect"][local_index]
            xe = coords[nodes]
            he = float(xe[1, 0] - xe[0, 0])
            area = float(block["element"]["properties"]["area"])

            if dtype == "BX":
                q = float(dload["value"]) * sign
            elif dtype == "GRAV":
                mat = materials[block["material"]]
                rho = float(mat["density"])
                q = rho * area * float(dload["value"]) * sign
            else:
                raise NotImplementedError(f"dload type {dtype!r} not supported for 1D")

            eft = [global_dof(n, j, dof_per_node) for n in nodes for j in range(dof_per_node)]
            qe = element_external_force_t1d1(q, he)
            F[eft] += qe


def collect_dirichlet_dofs(
    bcs: list[dict[str, Any]],
    dof_per_node: int,
) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    """Extract Dirichlet degrees of freedom and their prescribed values.

    This helper scans the boundary–condition list, selects entries of type
    :data:`DIRICHLET`, and converts their node lists and local degrees of
    freedom into global DOF indices.

    Returns
    -------
    tuple of (indices, values)
        ``indices`` is an integer array of global DOF numbers and
        ``values`` is a float array of the corresponding prescribed
        displacement values.
    """
    prescribed_dofs: list[int] = []
    prescribed_vals: list[float] = []

    for bc in bcs:
        if bc["type"] == DIRICHLET:
            for n in bc["nodes"]:
                I = global_dof(n, bc["local_dof"], dof_per_node)
                prescribed_dofs.append(I)
                prescribed_vals.append(float(bc["value"]))

    idx = np.array(prescribed_dofs, dtype=int)
    vals = np.array(prescribed_vals, dtype=float)
    return idx, vals


def solve_reduced_system(
    K: NDArray[np.float64],
    F: NDArray[np.float64],
    prescribed_idx: NDArray[np.int_],
    prescribed_vals: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve the constrained linear system and assemble the full DOF vector.

    The global system :math:`K u = F` is partitioned into free and prescribed
    degrees of freedom.  The reduced system for the free part is

    .. math::

        K_{ff} u_f = F_f - K_{fp} u_p ,

    where :math:`u_p` contains the prescribed values.  This routine

    1. builds the index sets of free and prescribed DOFs,
    2. forms :math:`K_{ff}` and :math:`K_{fp}`,
    3. solves for :math:`u_f`, and
    4. assembles and returns the full displacement vector ``u``.
    """
    num_dof = K.shape[0]
    all_dofs: NDArray[np.int_] = np.arange(num_dof, dtype=int)
    free_dofs: NDArray[np.int_] = np.setdiff1d(all_dofs, prescribed_idx)

    Kff = K[np.ix_(free_dofs, free_dofs)]
    Kfp = K[np.ix_(free_dofs, prescribed_idx)]
    Ff = F[free_dofs] - Kfp @ prescribed_vals
    uf = np.linalg.solve(Kff, Ff)

    dofs = np.zeros(num_dof, dtype=float)
    dofs[free_dofs] = uf
    dofs[prescribed_idx] = prescribed_vals
    return dofs


# ---------------------------------------------------------------------------
# Global solver
# ---------------------------------------------------------------------------


def first_fe_code(
    coords: NDArray[np.float64],
    blocks: list[dict[str, Any]],
    bcs: list[dict[str, Any]],
    dloads: list[dict[str, Any]],
    materials: dict[str, Any],
    block_elem_map: dict[int, tuple[int, int]],
) -> dict[str, Any]:
    """Assemble and solve a 1D linear–elastic finite element problem.

    This routine implements a small 1D bar solver for linear statics.  It uses
    preprocessed data from :mod:`wundy.ui` and performs the following steps:

    1. Assemble the global stiffness matrix from all element blocks using
       :func:`element_stiffness_t1d1` and the material response functions.
    2. Assemble the global right–hand side from Neumann boundary conditions
       via :func:`apply_neumann_bcs` and from distributed loads via
       :func:`apply_distributed_loads`.
    3. Impose Dirichlet boundary conditions by collecting prescribed degrees
       of freedom with :func:`collect_dirichlet_dofs` and solving the reduced
       linear system through :func:`solve_reduced_system`.
    4. Return the full displacement vector, global stiffness matrix and
       assembled force vector.

    The problem is a linear, static, 1D bar with one translational degree of
    freedom per node and two–node ``T1D1`` elements.

    Parameters
    ----------
    coords:
        Nodal coordinates with shape ``(num_node, num_dim)``; only the first
        column (x–coordinate) is used here.
    blocks:
        List of element–block dictionaries from :func:`wundy.ui.preprocess`.
        Each block must provide the connectivity (``"connect"``), element
        description (``"element"``) and material name (``"material"``).
    bcs:
        Boundary–condition dictionaries describing Dirichlet and Neumann
        conditions in preprocessed form.
    dloads:
        Distributed–load dictionaries describing body forces of type
        ``"BX"`` (user-specified line load) or ``"GRAV"`` (gravity using
        material density).
    materials:
        Mapping from material name to dictionaries that contain at least a
        ``"parameters"`` sub-dictionary with the Young’s modulus ``"E"`` and,
        for gravitational loads, a ``"density"`` entry.
    block_elem_map:
        Mapping from global element index to ``(block_index, local_elem_index)``
        specifying which block an element belongs to and its local index.

    Returns
    -------
    dict
        A dictionary with keys

        ``"dofs"`` :
            Full vector of nodal displacements.
        ``"stiff"`` :
            Global stiffness matrix :math:`K`.
        ``"force"`` :
            Global assembled force vector :math:`F`.
    """
    dof_per_node: int = 1
    num_node = coords.shape[0]
    num_dof = int(num_node * dof_per_node)

    K = np.zeros((num_dof, num_dof), dtype=float)
    F = np.zeros(num_dof, dtype=float)

    # Element stiffness assembly
    for block in blocks:
        area = float(block["element"]["properties"]["area"])
        material = materials[block["material"]]
        E = material_tangent_stiffness_elastic(material)

        for nodes in block["connect"]:
            eft = [global_dof(n, j, dof_per_node) for n in nodes for j in range(dof_per_node)]
            xe = coords[nodes]
            ke = element_stiffness_t1d1(xe, area, E)
            K[np.ix_(eft, eft)] += ke

    # External forces: Neumann + distributed loads
    apply_neumann_bcs(F, bcs, dof_per_node)
    apply_distributed_loads(F, dloads, coords, blocks, block_elem_map, materials, dof_per_node)

    # Dirichlet constraints and solution
    prescribed_idx, prescribed_vals = collect_dirichlet_dofs(bcs, dof_per_node)
    dofs = solve_reduced_system(K, F, prescribed_idx, prescribed_vals)

    solution: dict[str, Any] = {"dofs": dofs, "stiff": K, "force": F}
    return solution


def global_dof(node: int, local_dof: int, dof_per_node: int) -> int:
    """Return the global degree-of-freedom index for a node and local dof."""
    return node * dof_per_node + local_dof
