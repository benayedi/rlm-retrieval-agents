"""Write trajectories/README.md from scripts/docs/TRAJECTORIES.template.md and the per-row MANIFEST.json files.
usage: python scripts/docs/make_trajectories_readme.py"""
import json
from pathlib import Path

from naming import paper_names

ROOT = Path(__file__).resolve().parents[2]
NAME = {"snippet_repl": "Snippet REPL agent", "document_repl": "Document REPL agent"}
rows = []
for m in sorted((ROOT / "trajectories").glob("*/*/MANIFEST.json"), key=lambda p: (p.parent.name.split("-")[0], int(p.parent.name.split("-")[1]))):
    x = json.loads(m.read_text())
    trajs = sum(1 for f in x["files"] if "/traj/" in f["path"] or "/rollouts/" in f["path"] or f["path"].startswith("fulldoc/runs/")
                or (f["path"].endswith(".json") and f["path"].split("/")[-1].split(".")[0].isdigit()))
    arch = sum(p["bytes"] for p in x["archive_parts"])
    rel = m.parent.relative_to(ROOT / "trajectories").as_posix()
    rows.append(f"| [{x['row']}]({rel}/MANIFEST.json) | {NAME[x['agent']]} | {x['members']:,} | {x['member_bytes'] / 1e9:.2f} GB | "
                f"{arch / 1e9:.3f} GB ({len(x['archive_parts'])} part{'s' if len(x['archive_parts']) > 1 else ''}) | {'yes' if x['encrypted'] else 'no'} |")
table = ("| Row | Agent | Files | Original size | Archive (LFS) | Encrypted |\n|---|---|---|---|---|---|\n" + "\n".join(rows))
out = (ROOT / "scripts/docs/TRAJECTORIES.template.md").read_text().replace("{{TABLE}}", table)
(ROOT / "trajectories/README.md").write_text(paper_names(out))
print("wrote trajectories/README.md with", len(rows), "rows")
