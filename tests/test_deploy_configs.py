"""Static validation of deploy artifacts (no cluster needed; CI additionally runs kubeconform)."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_k8s_manifests_are_wired_consistently():
    docs = [d for d in yaml.safe_load_all((ROOT / "deploy/k8s/foundrate.yaml").read_text()) if d]
    kinds = {d["kind"] for d in docs}
    assert {"Namespace", "ConfigMap", "PersistentVolumeClaim", "Deployment", "Service"} <= kinds
    configmaps = {d["metadata"]["name"] for d in docs if d["kind"] == "ConfigMap"}
    claims = {d["metadata"]["name"] for d in docs if d["kind"] == "PersistentVolumeClaim"}
    for d in docs:
        assert d["apiVersion"] and d["metadata"]["name"]
        if d["kind"] != "Deployment":
            continue
        sel = d["spec"]["selector"]["matchLabels"]
        tpl = d["spec"]["template"]
        assert sel.items() <= tpl["metadata"]["labels"].items(), "selector must match pod labels"
        for v in tpl["spec"]["volumes"]:
            assert v["persistentVolumeClaim"]["claimName"] in claims
        for c in tpl["spec"]["containers"]:
            assert c["resources"]["limits"]
            for e in c.get("envFrom", []):
                assert e["configMapRef"]["name"] in configmaps
    svc = next(d for d in docs if d["kind"] == "Service")
    api = next(d for d in docs if d["kind"] == "Deployment" and d["metadata"]["name"] == "foundrate-api")
    assert svc["spec"]["selector"].items() <= api["spec"]["template"]["metadata"]["labels"].items()
    probe = api["spec"]["template"]["spec"]["containers"][0]["readinessProbe"]["httpGet"]["path"]
    assert probe == "/health"


def test_compose_and_prometheus_reference_existing_files():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    for svc in compose["services"].values():
        for vol in svc.get("volumes", []):
            if vol.startswith("./"):
                assert (ROOT / vol.split(":")[0]).exists(), vol
    prom = yaml.safe_load((ROOT / "deploy/prometheus/prometheus.yml").read_text())
    jobs = {j["job_name"]: j["static_configs"][0]["targets"][0] for j in prom["scrape_configs"]}
    assert jobs["foundrate-api"].split(":")[0] in compose["services"]
    assert jobs["foundrate-monitor"].split(":")[0] in compose["services"]


def test_grafana_dashboard_queries_only_exported_metrics():
    import json
    import re

    dash = json.loads((ROOT / "deploy/grafana/dashboards/foundrate.json").read_text())
    exported = ROOT.joinpath("foundrate/service/app.py").read_text() + ROOT.joinpath(
        "foundrate/monitoring/monitor.py").read_text()
    for p in dash["panels"]:
        for t in p["targets"]:
            for name in re.findall(r"foundrate_[a-z_0-9]+", t["expr"]):
                assert re.sub(r"_(bucket|count|sum|total)$", "", name) in exported or name in exported, name
