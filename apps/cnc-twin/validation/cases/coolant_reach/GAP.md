# coolant_reach: what is covered

Two measured cases (sources in db/bibliography/validation-4.yaml):

* `coolant-sultana2023-ti64-endmill-hpc-vs-dry`: tool-chip interface temperature (tool-work
  thermocouple) in end milling Ti-6Al-4V with a 12 mm HSS end mill, dry against one external 80 bar,
  2 L/min oil jet (2 mm nozzle, 20 mm from the edge) and a four-nozzle rotary applicator, full
  factorial over speed and feed (Sultana & Dhar, Heliyon 2023).
* `coolant-struzikiewicz2023-ti64-jet-direction`: chip size, chip-form grade and cutting force in
  finish turning Ti6Al4V ELI with one 150 bar emulsion jet (0.7 mm nozzle, 20 mm from the corner)
  aimed from nine directions at two feeds (Struzikiewicz, Materials 2023).

Related: the chip_form cases `chipform-slodki2019-...` and `chipform-zebala2020-...` give chip forms
with 70 bar coolant (and dry) for named Sandvik chip breakers.

Still missing: a flood-against-high-pressure comparison of tool or cutting-zone temperature in turning
with the nozzle position stated, and measured jet penetration or wetted area (high-speed imaging), e.g.
the CFD-and-imaging work on internal drill coolant (Oezkaya, Biermann) or Kramar/Krajnik/Kopac on
high-pressure jet-assisted turning; those sources were not reachable as open full text in this run.
