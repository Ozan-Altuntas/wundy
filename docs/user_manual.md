# Wundy User Manual — First Draft

## 1. Introduction

**Wundy** is a one-dimensional finite element (FE) solver for linear elastic bar problems.  
The code assembles the global stiffness matrix, applies boundary conditions, processes
distributed and concentrated loads, and returns the global displacement vector, stiffness
matrix, and force vector.

This manual describes:

- the physical problem solved by Wundy,
- the structure of a valid YAML input file,
- node, element, material, boundary-condition, and load syntax,
- preprocessing rules applied before assembly.

This draft intentionally focuses on the **input format** and the **conceptual problem
description**, as required for Project Week 1.

---

## 2. Physical Problem Description

Wundy solves **linear, static, one-dimensional bar problems** of the form:

\[
EA\,u'(x)' + q(x) = 0
\]

where:

- \( E \) — Young’s modulus  
- \( A \) — cross-sectional area  
- \( u(x) \) — axial displacement  
- \( q(x) \) — distributed load per unit length  

### Assumptions

The solver assumes:

1. **1D domain** with nodes on the real line.
2. **Two-node bar elements** (called `T1D1`).
3. **One degree of freedom per node** (axial displacement in the x-direction).
4. **Linear elastic materials**.
5. **Small-strain linear kinematics**.
6. **YAML input files** fully validated against the schema before assembly.

---

## 3. Overview of the Input File

A valid input file is a YAML document with one root key:

```yaml
wundy:
  nodes:
    - [node_id, x_coord]
    - [node_id, x_coord]

  elements:
    - [element_id, node1_id, node2_id]

  materials:
    - type: material_type
      name: material_name
      parameters:
        E: Youngs_modulus
        nu: Poissons_ratio
      density: material_density

  element blocks:
    - name: block_name
      material: material_name
      elements: element_set_name_or_ids
      element:
        type: T1D1
        properties:
          area: cross_section_area

  boundary conditions:
    - nodes: node_ids_or_node_set
      dof: x
      value: displacement_or_force
      type: DIRICHLET_or_NEUMANN

  node sets:
    - name: node_set_name
      nodes: [node_id, node_id]

  element sets:
    - name: element_set_name
      elements: [element_id, element_id]

  concentrated loads:
    - nodes: node_ids_or_node_set
      dof: x
      value: nodal_force_value

  distributed loads:
    - type: load_type   # BX or GRAV
      elements: element_set_or_ids
      value: load_magnitude
      direction: [±1.0]

---

## 4. Material Models

Wundy currently supports a single one-dimensional **linear elastic** material
model for bar elements. Additional constitutive models may be added in future
project installments, but all examples in this checkpoint use the elastic
model described below.

### 4.1 Elastic (type: `elastic`)

An elastic material is defined in the YAML input under the `materials` key as a
list of material objects. Each material has:

- a **type** (currently only `elastic`),
- a unique **name** used to reference it from element blocks,
- a `parameters` mapping containing the elastic constants, and
- an optional `density` used for gravitational body forces.

**YAML structure**

```yaml
materials:
  - type: elastic           # material model identifier
    name: mat-1             # user-defined material name
    parameters:
      E: 10.0               # Young's modulus (same units as stress)
      nu: 0.3               # Poisson's ratio (not used in 1D, kept for consistency)
    density: 1.0            # mass density (used for GRAV distributed loads)

