"""Interactive viewer: recording and HTML build."""
import json

from cad_d4d.losses.objective import ObjectiveConfig, ShapeObjective
from cad_d4d.optimization.srd import SRD, SRDConfig
from cad_d4d.visualization.recording import Recorder
from cad_d4d.visualization.web import build_html


def test_recorder_captures_frames_structures_and_events(reachable_target, coarse_sphere):
    obj = ShapeObjective(reachable_target, ObjectiveConfig())
    rec = Recorder(reachable_target, every=5, name="t")
    SRD(obj, SRDConfig(rounds=3, steps_per_round=10, polish_steps=5)).run(coarse_sphere, recorder=rec)
    d = json.loads(json.dumps(rec.to_dict()))  # JSON round trip
    steps = [f["step"] for f in d["frames"]]
    assert steps[0] == 0 and steps == sorted(steps) and steps[-1] == 35
    assert len(d["structures"]) >= 1
    s = d["structures"][d["frames"][-1]["structure"]]
    assert len(s["dof_kinds"]) == s["n_cp"]
    assert any(e["type"] == "proposal" for e in d["events"])


def test_build_html_embeds_data_safely():
    rows = [{"target": "t", "tag": "x", "method": "uniform_k0", "kind": "fixed", "fit": 1.0, "n_cp": 56}]
    html = build_html([{"name": "</script>evil", "frames": []}], rows, title="T")
    assert "__D4D_DATA__" not in html
    assert "</script>evil" not in html  # closing tags inside data are escaped
    payload = html.split('id="d4d-data" type="application/json">')[1].split("</script>")[0]
    data = json.loads(payload.replace("<\\/", "</"))
    assert data["benchmark"]["rows"][0]["method"] == "uniform_k0"
