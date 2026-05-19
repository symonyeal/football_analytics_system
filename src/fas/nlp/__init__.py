"""Nonlinear programming module (Part 3): EPV, pass value, trajectory, set piece.

All four submodules require optional heavy dependencies (torch / cyipopt /
cma); they are documented stubs with full formulations. Import them directly
when their extras are installed:

    fas.nlp.epv_unet         (extras: ml)   U-Net EPV surface (Part 3.1)
    fas.nlp.pass_value_nlp   (extras: nlp)  SQP + CMA-ES pass optimizer (3.2)
    fas.nlp.trajectory_opt   (extras: nlp)  IPOPT direct collocation (3.3)
    fas.nlp.set_piece_opt    (extras: ml)   Magnus-force trajectory NLP (3.4)
"""

__all__: list[str] = []
