"""Holding domain: tool holding and balance, and workholding (docs/models/holding.md).

Tool holders: clamping torque and axial retention per holder type (ER collets through the nut thread,
collar and 8 deg cone; hydraulic and shrink-fit chucks from the catalogue torque or the Lame interference
pressure; milling chucks; side-lock positive drive with axial walk; floating tapping holders) against the
cutting torque and axial pull, with slip and pull-out integrated as a 2-DOF Coulomb ratchet; tool and
holder unbalance (ISO 21940-11) as a static vector and couple in the rotating tool frame; speed limits.

Workholding: every device of a setup becomes a set of elastic contacts with Coulomb friction on the part
(vises, strap clamps, fixture plates with locators, soft jaws, vacuum, magnetic and adhesive holding, 3- and
4-jaw chucks with the centrifugal jaw loss, collets, tailstock centres, steady rests, tabs); each macro step
the cutting wrench (mean and peak per tick), gravity and inertia are checked for slip, lift-off, tipping,
rotation and pull-out, the part pose is updated when it slips, the part becomes free when its holding is
lost, and the clamp loads go to structure. Chips and burrs under locators give the location error.

The domain object is :class:`cnctwin.holding.domain.HoldingDomain`; its geometry protocols are in
:mod:`cnctwin.holding.protocols`.
"""
