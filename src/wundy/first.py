from typing import Any

import numpy as np
from numpy.typing import NDArray

from .schemas import DIRICHLET
from .schemas import NEUMANN


def first_fe_code(
    coords: NDArray[np.float64],
    blocks: list[dict[str, Any]],
    bcs: list[dict[str, Any]],
    dloads: list[dict[str, Any]],
    materials: dict[str, Any],
    block_elem_map: dict[int, tuple[int, int]],
) -> dict[str, Any]:
    """Assemble and solve a 1D linear–elastic finite element problem.

    This routine implements a small 1D bar solver for linear statics. It expects
    preprocessed input produced by :mod:`wundy.ui` and performs the following steps:

    1. Assemble the global stiffness matrix from all element blocks.
    2. Assemble the global right–hand side from Neumann boundary conditions
       and distributed loads (body forces).
    3. Impose Dirichlet boundary conditions by eliminating prescribed
       degrees of freedom in a symmetry-preserving way.
    4. Solve the reduced linear system for the free degrees of freedom and
       recover the full displacement vector.

    Parameters
    ----------
    coords:
        Array of nodal coordinates with shape ``(num_node, num_dim)``.
        Only the first coordinate (``x``) is used for the 1D formulation.
    blocks:
        List of element-block dictionaries created by :func:`wundy.ui.preprocess`.
        Each block must at least contain the keys ``"connect"``, ``"element"``,
        and ``"material"``.
    bcs:
        List of boundary-condition dictionaries from :func:`wundy.ui.preprocess`.
        Each entry stores the boundary type (Dirichlet or Neumann), the local
        degree of freedom, the node indices, and the prescribed value.
    dloads:
        List of distributed-load dictionaries from :func:`wundy.ui.preprocess`.
        Supported load types are ``"BX"`` (user-specified line load) and
        ``"GRAV"`` (gravitational body force using the material density).
    materials:
        Mapping from material name to dictionaries with at least a
        ``"parameters"`` sub-dictionary containing the Young’s modulus
        ``"E"``. For gravitational loads the material dictionary may also
        contain ``"density"``.
    block_elem_map:
        Mapping from global element index to ``(block_index, local_elem_index)``
        indicating which block an element belongs to and its local index within
        that block.

    Returns
    -------
    dict
        A dictionary with the following keys:

        ``"dofs"`` :
            One-dimensional array of nodal displacement degrees of freedom.
        ``"stiff"`` :
            Global stiffness matrix :math:`K`.
        ``"force"`` :
            Global assembled force vector :math:`F`.

    Notes
    -----
    Assumptions of this implementation:

    * One translational degree of freedom per node (axial displacement).
    * Two-node linear bar elements (``T1D1``) with constant cross-sectional area.
    * Small-strain, linear-elastic behavior.
    * All input has been validated and preprocessed by :mod:`wundy.ui`.
    """
    dof_per_node: int = 1
    num_node = coords.shape[0]
    num_dof = int(num_node * dof_per_node)

    K = np.zeros((num_dof, num_dof), dtype=float)
    F = np.zeros(num_dof, dtype=float)

    # Assemble global stiffness
    for block in blocks:
        A = block["element"]["properties"]["area"]
        material = materials[block["material"]]
        E = material["parameters"]["E"]
        for nodes in block["connect"]:
            eft = [global_dof(n, j, dof_per_node) for n in nodes for j in range(dof_per_node)]
            xe = coords[nodes]
            he = float(xe[1, 0] - xe[0, 0])
            if np.isclose(he, 0.0):
                raise ValueError(f"Zero-length element detected between nodes {nodes}")
            ke = A * E / he * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
            K[np.ix_(eft, eft)] += ke

    # Apply Neumann boundary conditions to force
    for bc in bcs:
        if bc["type"] == NEUMANN:
            for n in bc["nodes"]:
                I = global_dof(n, bc["local_dof"], dof_per_node)
                F[I] += bc["value"]

    # Apply distributed loads
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
            A = block["element"]["properties"]["area"]

            if dtype == "BX":
                q = dload["value"] * sign
            elif dtype == "GRAV":
                mat = materials[block["material"]]
                rho = mat["density"]
                q = rho * A * dload["value"] * sign
            else:
                raise NotImplementedError(f"dload type {dtype!r} not supported for 1D")

            eft = [global_dof(n, j, dof_per_node) for n in nodes for j in range(dof_per_node)]
            qe = q * he / 2.0 * np.ones(2, dtype=float)
            F[eft] += qe

    # Collect prescribed dofs (Dirichlet)
    prescribed_dofs: list[int] = []
    prescribed_vals: list[float] = []
    for bc in bcs:
        if bc["type"] == DIRICHLET:
            for n in bc["nodes"]:
                I = global_dof(n, bc["local_dof"], dof_per_node)
                prescribed_dofs.append(I)
                prescribed_vals.append(float(bc["value"]))

    all_dofs: NDArray[np.int_] = np.arange(num_dof, dtype=int)
    prescribed_idx: NDArray[np.int_] = np.array(prescribed_dofs, dtype=int)
    free_dofs: NDArray[np.int_] = np.setdiff1d(all_dofs, prescribed_idx)

    Kff = K[np.ix_(free_dofs, free_dofs)]
    Kfp = K[np.ix_(free_dofs, prescribed_idx)]
    Ff = F[free_dofs] - Kfp @ np.array(prescribed_vals, dtype=float)
    uf = np.linalg.solve(Kff, Ff)

    # Solve the system and assemble full dof vector
    dofs = np.zeros(num_dof, dtype=float)
    dofs[free_dofs] = uf
    dofs[prescribed_idx] = np.array(prescribed_vals, dtype=float)

    solution: dict[str, Any] = {"dofs": dofs, "stiff": K, "force": F}
    return solution


def global_dof(node: int, local_dof: int, dof_per_node: int) -> int:
    """Return the global degree-of-freedom index for a node and local dof.

    Parameters
    ----------
    node:
        Zero-based node index in the global mesh.
    local_dof:
        Local degree-of-freedom index at the node (0 for axial displacement
        in this 1D solver).
    dof_per_node:
        Number of degrees of freedom stored per node in the global system.

    Returns
    -------
    int
        The global degree-of-freedom index suitable for indexing global
        vectors and matrices.
    """
    return node * dof_per_node + local_dof
