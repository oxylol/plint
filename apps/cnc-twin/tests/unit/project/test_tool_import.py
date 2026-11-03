import pytest

from cnctwin.core.errors import InputError
from cnctwin.project import import_iso13399, import_tools_csv, parse_p21, resolve
from cnctwin.project.iso13399 import Enum, Ref, Typed, decode_string, tool_items
from conftest import PROJECT, write

CSV = """\
# tool list exported from the presetter
number,tool,holder,stickout_mm,length_offset_mm,radius_offset_mm,comment,diameter_mm
1,test-em-10,test-er32,35,120.53,5.0,"D10 4FL, carbide",
2,test-em-6,test-er32,25,98.2,,,5.97
"""

CSV_INLINE = """\
number,holder,stickout_mm,type,hand,tool_material,tool_material_class,coating,diameter_mm,flutes,helix_deg,rake_deg,cutting_length_mm,overall_length_mm,shank_diameter_mm
3,test-er32,30,square_endmill,right,test-carbide,carbide,none,8,3,40,10,16,63,8
"""

# encoding 1: symbol and measure in one instance; encoding 2: symbol, description, number, unit;
# encoding 3: a property-definition chain; units of the file: millimetre and degree
P21_MM = """\
ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('ISO 13399 tool data'),'2;1');
FILE_NAME('em.p21','2026-11-03T10:00:00',('test'),(''),'','','');
FILE_SCHEMA(('ISO13399'));
ENDSEC;
DATA;
#1=PRODUCT('2P342-1000-PA','Solid carbide end mill, 4 flutes','',(#2));
#2=PRODUCT_CONTEXT('',#3,'mechanical');
#3=APPLICATION_CONTEXT('tool data');
#4=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.));
#5=(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.));
#6=(CONVERSION_BASED_UNIT('DEGREE',#7)NAMED_UNIT(*)PLANE_ANGLE_UNIT());
#7=PLANE_ANGLE_MEASURE_WITH_UNIT(PLANE_ANGLE_MEASURE(0.0174532925),#5);
#10=PROPERTY_VALUE('DC',LENGTH_MEASURE(10.0),#4);
#11=PROPERTY_VALUE('APMX',LENGTH_MEASURE(22.),#4);
#12=PROPERTY('OAL','overall length',72.,'mm');
#13=PROPERTY('DMM','shank diameter',10.,'mm');
#14=PROPERTY('ZEFP','peripheral effective cutting edge count',4.,'');
#15=PROPERTY('FHA','flute helix angle',38.,'deg');
#16=PROPERTY_DEFINITION('GAMF','radial rake angle',#1);
#17=PROPERTY_DEFINITION_REPRESENTATION(#16,#18);
#18=REPRESENTATION('',(#19),#3);
#19=MEASURE_REPRESENTATION_ITEM('value',PLANE_ANGLE_MEASURE(8.),#6);
#20=PROPERTY('RE','corner radius',0.,'mm');
#21=PROPERTY('BMC','body material code','HC');
#22=PROPERTY('HAND','hand','R');
#23=PROPERTY('WT','weight of item',0.04,'kg');
#24=PROPERTY('RPMX','rotational speed maximum',40000.,'rpm');
#25=PROPERTY('CCMS','connection code machine side','C-ZYL');
/* a comment that the parser skips */
ENDSEC;
END-ISO-10303-21;
"""

P21_INCH = """\
ISO-10303-21;
HEADER;
FILE_DESCRIPTION((''),'2;1');
ENDSEC;
DATA;
#1=PRODUCT('DRILL-0250','Jobber drill \\X2\\00D8\\X0\\ 1/4 in','',());
#2=(CONVERSION_BASED_UNIT('INCH',#3)LENGTH_UNIT()NAMED_UNIT(*));
#3=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(25.4),#4);
#4=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.));
#10=PROPERTY_VALUE('DC',LENGTH_MEASURE(0.25),#2);
#11=PROPERTY_VALUE('OAL',LENGTH_MEASURE(4.),#2);
#12=PROPERTY_VALUE('LCF',LENGTH_MEASURE(2.75),#2);
#13=PROPERTY_VALUE('SIG',PLANE_ANGLE_MEASURE(118.),$);
ENDSEC;
END-ISO-10303-21;
"""


def test_csv_rows_become_tool_list_entries(tmp_path):
    path = write(tmp_path / "tools.csv", CSV)
    entries, lines, problems = import_tools_csv(path)
    assert problems == []
    assert entries[0] == {"number": 1, "tool": "test-em-10", "holder": "test-er32", "stickout_mm": 35.0,
                          "length_offset_mm": 120.53, "radius_offset_mm": 5.0, "comment": "D10 4FL, carbide"}
    assert entries[1]["tool"] == {"geometry": {"diameter_mm": 5.97}, "extends": "test-em-6", "id": "tools-t2"}
    assert lines == {0: 3, 1: 4}


def test_csv_problems_name_line_and_column(tmp_path):
    entries, lines, problems = import_tools_csv(write(tmp_path / "a.csv", "number,tool,colour\n1,x,red\n"))
    assert problems[0].path == "colour" and "unknown column" in problems[0].message and problems[0].line == 1
    entries, lines, problems = import_tools_csv(write(tmp_path / "b.csv", "number,tool,stickout_mm\n1,x,long\n"))
    assert problems[0].line == 2 and "'long' is not a number" in problems[0].message
    entries, lines, problems = import_tools_csv(write(tmp_path / "c.csv", "number,stickout_mm\n1,30\n"))
    assert "needs a tool id" in problems[0].message


def test_csv_library_in_a_project(project_dir, fixture_db):
    write(project_dir / "tools.csv", CSV_INLINE)
    text = PROJECT.replace("  library: null", "  library: tools.csv")
    (project_dir / "project.yaml").write_text(text)
    p = resolve(project_dir, fixture_db)
    t3 = p.tools.list[2]
    assert t3.number == 3 and t3.tool.geometry.diameter_m == pytest.approx(0.008) and t3.tool.src == "project"
    assert t3.tool.tool_material.id == "test-carbide" and "tools.csv" in p.input_hashes
    write(project_dir / "tools.csv", CSV_INLINE.replace("square_endmill", "square_endmil"))
    with pytest.raises(InputError) as info:
        resolve(project_dir, fixture_db)
    assert "square_endmil" in str(info.value) and "did you mean 'square_endmill'" in str(info.value)
    write(project_dir / "tools.csv", "number,tool,holder\n1,test-em-6,test-er32\n")
    with pytest.raises(InputError) as info:
        resolve(project_dir, fixture_db)
    assert "tool number 1 is also in tools.list" in str(info.value)


def test_p21_exchange_structure():
    p21 = parse_p21(P21_MM, "em.p21")
    assert p21.header["FILE_SCHEMA"] == [["ISO13399"]]
    assert p21.instances[10].entity == "PROPERTY_VALUE"
    assert p21.instances[10].params == ["DC", Typed("LENGTH_MEASURE", 10.0), Ref(4)]
    assert p21.instances[4].entity == "" and [e for e, _ in p21.instances[4].parts] == ["LENGTH_UNIT", "NAMED_UNIT",
                                                                                          "SI_UNIT"]
    assert p21.instances[4].parts[2][1] == [Enum("MILLI"), Enum("METRE")]
    assert decode_string("D\\X2\\00D8\\X0\\ 6 it''s") == "DØ 6 it''s"
    with pytest.raises(InputError) as info:
        parse_p21("ISO-10303-21;\nDATA;\n#1=X(1,;\n", "bad.p21")
    assert "bad.p21:3" in str(info.value)


def test_iso13399_symbols_map_to_template_fields():
    item = tool_items(parse_p21(P21_MM, "em.p21"))[0]
    assert item["id"] == "2p342-1000-pa" and item["order_code"] == "2P342-1000-PA"
    assert item["type"] == "square_endmill" and item["hand"] == "right"
    assert item["tool_material"] == {"class": "carbide"}
    geometry = item["geometry"]
    assert geometry["diameter_mm"] == 10.0 and geometry["cutting_length_mm"] == 22.0
    assert geometry["overall_length_mm"] == 72.0 and geometry["shank"] == {"diameter_mm": 10.0}
    assert geometry["flutes"] == 4 and geometry["helix_deg"] == 38.0
    assert geometry["rake_deg"] == pytest.approx(8.0)              # through the property-definition chain
    assert geometry["corner_radius_mm"] == 0.0
    assert item["mass_g"] == pytest.approx(40.0) and item["max_rpm"] == 40000.0
    assert "CCMS" in item["notes"]                                    # not in the mapping table: listed


def test_iso13399_file_units(tmp_path):
    item = tool_items(parse_p21(P21_INCH, "drill.p21"))[0]
    assert item["type"] == "twist_drill" and item["id"] == "drill-0250"
    assert item["geometry"]["diameter_mm"] == pytest.approx(6.35)
    assert item["geometry"]["overall_length_mm"] == pytest.approx(101.6)
    assert item["geometry"]["flute_length_mm"] == pytest.approx(69.85)
    assert item["geometry"]["point_angle_deg"] == pytest.approx(118.0)
    write(tmp_path / "lib" / "a.p21", P21_MM)
    write(tmp_path / "lib" / "b.stp", P21_INCH)
    assert [i["id"] for i in import_iso13399(tmp_path / "lib")] == ["2p342-1000-pa", "drill-0250"]


def test_iso13399_library_in_a_project(project_dir, fixture_db):
    write(project_dir / "iso13399" / "em.p21", P21_MM)
    text = PROJECT.replace("  library: null", "  library: iso13399").replace(
        "tool: test-em-10,", "tool: {extends: 2p342-1000-pa, tool_material: {class: carbide, id: test-carbide}, "
                             "coating: {id: test-tialn, thickness_um: 3}},")
    (project_dir / "project.yaml").write_text(text)
    p = resolve(project_dir, fixture_db)
    tool = p.tools.list[0].tool
    assert tool.geometry.diameter_m == pytest.approx(0.01) and tool.coating.id == "test-tialn"
    assert tool.geometry.helix_rad == pytest.approx(0.663225, rel=1e-5)
    assert "iso13399/em.p21" in p.input_hashes
    missing = PROJECT.replace("  library: null", "  library: iso13399").replace("tool: test-em-10,",
                                                                                "tool: 2p342-1000-pa,")
    (project_dir / "project.yaml").write_text(missing)
    with pytest.raises(InputError) as info:
        resolve(project_dir, fixture_db)
    assert "missing required key 'id'" in str(info.value) or "tool_material" in str(info.value)
