"""Smoke test of the report's heatmap figures on an ablation checkpoint (CPU)."""
import importlib.util, os, sys
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("rep", os.path.join(os.path.dirname(__file__), "09_report.py")); rep = importlib.util.module_from_spec(spec); spec.loader.exec_module(rep)
h = rep.figs(f"{rep.R}/abl_lr2e4/model.pt")
print(len(h)); print(h[:600])
