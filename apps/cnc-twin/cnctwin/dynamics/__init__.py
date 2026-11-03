"""Dynamics domain: machine dynamics and vibration (docs/models/dynamics.md).

Modal systems per controller path (tool modes from structure or the spindle / turret modes, machine frame, table and
rotary-axis modes, workpiece modes at the contact), integrated exactly by first-order hold with a predictor-corrector
in contact (cnctwin.dynamics.kernels) and exactly with the harmonic unbalance and first-order-hold base excitation
out of contact; exact energy integrals for the ledger; chatter detection from the vibration samples
(cnctwin.dynamics.chatter); stability lobe diagrams (cnctwin.dynamics.sld: zero-order, multi-frequency, variable
pitch, turning, mode coupling, process damping) checked by semi-discretisation (cnctwin.dynamics.semidisc).

The domain object is :class:`cnctwin.dynamics.domain.DynamicsDomain`.
"""
