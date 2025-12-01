import io

import numpy as np

import wundy
import wundy.first


def test_first_1():
    file = io.StringIO()
    file.write("""\
wundy:
  nodes: [[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]
  elements: [[1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 5]]
  boundary conditions:
  - name: fix-nodes
    dof: x
    nodes: [1]
  concentrated loads:
  - name: cload-1
    nodes: [5]
    value: 2.0
  materials:
  - type: elastic
    name: mat-1
    parameters:
      E: 10.0
      nu: 0.3
  element blocks:
  - material: mat-1
    name: block-1
    elements: all
    element:
      type: t1d1
      properties:
        area: 1
""")
    file.seek(0)
    data = wundy.ui.load(file)
    inp = wundy.ui.preprocess(data)
    soln = wundy.first.first_fe_code(
        inp["coords"],
        inp["blocks"],
        inp["bcs"],
        inp["dload"],
        inp["materials"],
        inp["block_elem_map"],
    )

    dofs = soln["dofs"]
    K = soln["stiff"]
    F = soln["force"]
    assert np.allclose(dofs, [0, 0.2, 0.4, 0.6, 0.8])
    assert np.allclose(F, [0, 0, 0, 0, 2])
    assert np.allclose(
        K,
        [
            [10, -10, 0, 0, 0],
            [-10, 20, -10, 0, 0],
            [0, -10, 20, -10, 0],
            [0, 0, -10, 20, -10],
            [0, 0, 0, -10, 10],
        ],
    )


def test_first_2():
    """Bar loaded by a uniform axial distributed load (BX).

    The model is the same 4-element bar as in ``test_first_1``:

    * Nodes at x = 0, 1, 2, 3, 4
    * Element connectivity: [1–2], [2–3], [3–4], [4–5]
    * Left end (node 1) fixed in x
    * Uniform line load q = 2.0 applied to all elements in the +x direction

    The expected stiffness matrix is unchanged from ``test_first_1``, while the
    equivalent nodal force vector and displacements are different.  The exact
    reference values below were obtained by solving the same assembled system
    analytically.
    """
    file = io.StringIO()
    file.write("""\
wundy:
  nodes: [[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]
  elements: [[1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 5]]
  boundary conditions:
  - name: fix-nodes
    dof: x
    nodes: [1]
  materials:
  - type: elastic
    name: mat-1
    parameters:
      E: 10.0
      nu: 0.3
  distributed loads:
  - name: dload-1
    type: BX
    elements: [1, 2, 3, 4]
    value: 2.0
    direction: [1.0]
  element blocks:
  - material: mat-1
    name: block-1
    elements: ALL
    element:
      type: t1d1
      properties:
        area: 1
""")
    file.seek(0)
    data = wundy.ui.load(file)
    inp = wundy.ui.preprocess(data)
    soln = wundy.first.first_fe_code(
        inp["coords"],
        inp["blocks"],
        inp["bcs"],
        inp["dload"],
        inp["materials"],
        inp["block_elem_map"],
    )

    dofs = soln["dofs"]
    K = soln["stiff"]
    F = soln["force"]

    # Displacements resulting from the uniform line load q = 2.0
    assert np.allclose(dofs, [0.0, 0.7, 1.2, 1.5, 1.6])

    # Equivalent nodal forces from the distributed load
    assert np.allclose(F, [1.0, 2.0, 2.0, 2.0, 1.0])

    # Stiffness matrix is the same as in the point-load case
    assert np.allclose(
        K,
        [
            [10, -10, 0, 0, 0],
            [-10, 20, -10, 0, 0],
            [0, -10, 20, -10, 0],
            [0, 0, -10, 20, -10],
            [0, 0, 0, -10, 10],
        ],
    )

def test_gauss_element_stiffness_t1d1():
    """element_stiffness_t1d1 should match the analytic bar stiffness matrix.

    For a 2-node bar of length h = 1, area A = 2, and E = 10, the exact
    stiffness matrix is

        k = (E*A/h) * [[1, -1],
                       [-1, 1]].

    The Gauss-quadrature implementation in element_stiffness_t1d1 should
    reproduce this exactly.
    """
    # Element from x = 0 to x = 1
    xe = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    area = 2.0
    E = 10.0
    h = 1.0

    ke = wundy.first.element_stiffness_t1d1(xe, area, E)

    ke_exact = (E * area / h) * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
    assert np.allclose(ke, ke_exact)

def test_gauss_element_internal_force_t1d1():
    """element_internal_force_t1d1 should reproduce the exact internal forces.

    For a bar of length h = 1, A = 2, E = 10 with nodal displacements
    u = [0.0, 0.1], the strain is 0.1 and the stress is 1.0. The exact
    internal nodal forces are [-2.0, 2.0].
    """
    xe = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    area = 2.0
    material = {
        "parameters": {"E": 10.0, "nu": 0.3},
    }
    ue = np.array([0.0, 0.1], dtype=float)

    fint = wundy.first.element_internal_force_t1d1(xe, area, material, ue)

    fint_exact = np.array([-2.0, 2.0], dtype=float)
    assert np.allclose(fint, fint_exact)

def test_gauss_points_1d_two_point():
    """Check that the 2-point Gauss rule is implemented correctly."""
    xi, w = wundy.first.gauss_points_1d_two_point()

    # Points should be ±1/sqrt(3), weights should be 1.
    expected_xi = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)], dtype=float)
    expected_w = np.ones(2, dtype=float)

    assert np.allclose(xi, expected_xi)
    assert np.allclose(w, expected_w)
