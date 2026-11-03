"""Wear domain: tool wear, damage, life and breakage (docs/models/wear.md).

Per tool slot and edge element: flank wear by Usui's law at every in-cut quadrature temperature
(published constants, or calibrated to the work material's Taylor life), crater, notch at the
depth-of-cut line, abrasive (Archard) wear, coating first and substrate when worn through, BUE
protection and attrition, edge rounding; Weibull chipping per entry shock with counter-based draws
and coating spallation; thermal-cycle fatigue and comb cracks; section breakage by overload, packing
loads and Basquin fatigue; life criteria, usage counters and the remaining life.

The domain object is :class:`cnctwin.wear.domain.WearDomain`; its protocols are in
:mod:`cnctwin.wear.protocols` and :mod:`cnctwin.wear.calibration`.
"""
