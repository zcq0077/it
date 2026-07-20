"""Validation-only multi-stage hyperparameter tuning for iTentformer.

The search never evaluates the test split. After all validation trials finish,
an optional eval-only process evaluates the single winning checkpoint once.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
MAIN_SCRIPT = ROOT / "iTentformer.py"
METRIC_PATTERN = re.compile(
    r"(?P<stage>Final Validation|Final Test), run \d+/\d+, epoch (?P<epoch>\d+), "
    r"loss (?P<loss>[0-9.eE+-]+), ADE (?P<ade>[0-9.eE+-]+)nmi .*?"
    r"FDE (?P<fde>[0-9.eE+-]+)nmi .*?RMSE_COG (?P<cog>[0-9.eE+-]+)deg, "
    r"RMSE_SOG (?P<sog>[0-9.eE+-]+)kn\."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable validation-only hyperparameter search."
    )
    parser.add_argument("--base-config", default="config_iTentformer.py")
    parser.add_argument("--study-name", default="dma_v15_auto_tune")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--screen-epochs", type=int, default=24)
    parser.add_argument("--screen-stride", type=int, default=4)
    parser.add_argument("--screen-patience", type=int, default=6)
    parser.add_argument("--screen-warmup", type=int, default=5)
    parser.add_argument("--max-screen-trials", type=int, default=8)
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--final-epochs", type=int, default=50)
    parser.add_argument("--final-patience", type=int, default=12)
    parser.add_argument(
        "--run-final-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate the winning full-resolution checkpoint on the test set once.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_python_config(path: Path) -> dict:
    module_name = f"auto_tune_base_{abs(hash(str(path.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "get_config"):
        config = module.get_config()
    elif hasattr(module, "config") and hasattr(module.config, "to_dict"):
        config = module.config.to_dict()
    else:
        raise RuntimeError(f"Config does not expose get_config()/config.to_dict(): {path}")
    if not isinstance(config, dict):
        raise TypeError("Base config must resolve to a dictionary.")
    return config


def parse_metric(log_path: Path, stage: str) -> dict | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = [m for m in METRIC_PATTERN.finditer(text) if m.group("stage") == stage]
    if not matches:
        return None
    values = matches[-1].groupdict()
    metric = {
        "stage": stage,
        "epoch": int(values["epoch"]),
        "loss": float(values["loss"]),
        "ade": float(values["ade"]),
        "fde": float(values["fde"]),
        "rmse_cog": float(values["cog"]),
        "rmse_sog": float(values["sog"]),
    }
    return metric


def validation_objective(metric: dict, fde_weight: float) -> float:
    return metric["ade"] + fde_weight * metric["fde"]


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


def log_message(path: Path, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def save_leaderboard(path: Path, records: list[dict]) -> None:
    completed = [
        record
        for record in records
        if record.get("status") == "completed" and record.get("validation")
    ]
    completed.sort(key=lambda item: item["objective"])
    fields = [
        "rank",
        "trial_id",
        "phase",
        "objective",
        "ade_nmi",
        "fde_nmi",
        "best_epoch",
        "learning_rate",
        "lr_scheduler_patience",
        "lr_min",
        "subroute_residual_scale",
        "balanced_intent_loss_weight",
        "window_stride",
        "epochs",
        "checkpoint",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, record in enumerate(completed, start=1):
            params = record["params"]
            metric = record["validation"]
            writer.writerow({
                "rank": rank,
                "trial_id": record["trial_id"],
                "phase": record["phase"],
                "objective": f"{record['objective']:.6f}",
                "ade_nmi": f"{metric['ade']:.6f}",
                "fde_nmi": f"{metric['fde']:.6f}",
                "best_epoch": metric["epoch"],
                "learning_rate": params["learning_rate"],
                "lr_scheduler_patience": params["lr_scheduler_patience"],
                "lr_min": params["lr_min"],
                "subroute_residual_scale": params["subroute_residual_scale"],
                "balanced_intent_loss_weight": params["balanced_intent_loss_weight"],
                "window_stride": params["window_stride"],
                "epochs": params["epochs"],
                "checkpoint": record["checkpoint"],
            })


def screen_candidates(base: dict) -> list[dict]:
    common = {
        "learning_rate": 2e-4,
        "lr_scheduler": "plateau",
        "lr_reduce_factor": 0.5,
        "lr_scheduler_patience": 5,
        "lr_min": 3.125e-6,
        "subroute_residual_scale": 0.25,
        "balanced_intent_loss_weight": 0.35,
    }
    candidates = [
        ("baseline", {}),
        ("lr_pat3", {"lr_scheduler_patience": 3}),
        ("lr_pat7_floor6", {"lr_scheduler_patience": 7, "lr_min": 6.25e-6}),
        ("expert015", {"subroute_residual_scale": 0.15}),
        ("expert020", {"subroute_residual_scale": 0.20}),
        ("balance020", {"balanced_intent_loss_weight": 0.20}),
        ("balance025", {"balanced_intent_loss_weight": 0.25}),
        (
            "expert020_balance025",
            {
                "subroute_residual_scale": 0.20,
                "balanced_intent_loss_weight": 0.25,
            },
        ),
    ]
    result = []
    for name, overrides in candidates:
        params = dict(common)
        params.update(overrides)
        result.append({"name": name, "params": params})
    return result


class Study:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.base_config_path = resolve_path(args.base_config)
        self.base = load_python_config(self.base_config_path)
        self.root = ROOT / "tuning_results" / args.study_name
        self.config_dir = self.root / "configs"
        self.run_dir = self.root / "runs"
        self.model_dir = self.root / "checkpoints"
        self.console_dir = self.root / "console"
        self.state_path = self.root / "study.json"
        self.pipeline_log = self.root / "pipeline.log"
        self.leaderboard_path = self.root / "leaderboard.csv"
        self.lock_path = self.root / "RUNNING.lock"
        self.root.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(exist_ok=True)
        self.run_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        self.console_dir.mkdir(exist_ok=True)
        if args.resume and self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        else:
            self.state = {
                "study_name": args.study_name,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "base_config": str(self.base_config_path),
                "objective": "validation_ADE + early_stop_fde_weight * validation_FDE",
                "trials": [],
            }

    def save(self) -> None:
        self.state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        atomic_write_json(self.state_path, self.state)
        save_leaderboard(self.leaderboard_path, self.state["trials"])

    def existing(self, trial_id: str) -> dict | None:
        for record in self.state["trials"]:
            if record["trial_id"] == trial_id:
                return record
        return None

    def build_config(
        self,
        trial_id: str,
        phase: str,
        params: dict,
        epochs: int,
        stride: int,
        patience: int,
        selector_warmup: int,
    ) -> tuple[dict, Path, Path, Path]:
        config = dict(self.base)
        config.update(params)
        config.update({
            "epochs": epochs,
            "window_stride": stride,
            "patience": patience,
            "candidate_selector_warmup_epochs": selector_warmup,
            "evaluate_test_each_epoch": False,
            "evaluate_final_test": False,
            "plot_count": 0,
            "eval_only": False,
            "checkpoint_path": None,
            "append_log": False,
            "results_dir": str(self.run_dir),
            "run_name": trial_id,
            "model_dir": str(self.model_dir),
            "model_prefix": trial_id,
        })
        config_path = self.config_dir / f"{trial_id}.json"
        train_log = self.run_dir / trial_id / "train.log"
        checkpoint = self.model_dir / f"{trial_id}_fixed.pt"
        atomic_write_json(config_path, config)
        return config, config_path, train_log, checkpoint

    def run_trial(
        self,
        trial_id: str,
        phase: str,
        params: dict,
        epochs: int,
        stride: int,
        patience: int,
        selector_warmup: int,
    ) -> dict:
        existing = self.existing(trial_id)
        if existing and existing.get("status") == "completed":
            log_message(self.pipeline_log, f"Resume: skip completed {trial_id}.")
            return existing

        config, config_path, train_log, checkpoint = self.build_config(
            trial_id,
            phase,
            params,
            epochs,
            stride,
            patience,
            selector_warmup,
        )
        effective_params = {
            key: config[key]
            for key in (
                "learning_rate",
                "lr_scheduler",
                "lr_reduce_factor",
                "lr_scheduler_patience",
                "lr_min",
                "subroute_residual_scale",
                "balanced_intent_loss_weight",
                "window_stride",
                "epochs",
                "train_seed",
            )
        }
        record = existing or {
            "trial_id": trial_id,
            "phase": phase,
            "params": effective_params,
            "config": str(config_path),
            "checkpoint": str(checkpoint),
        }
        record.update({
            "status": "planned" if self.args.dry_run else "running",
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "validation": None,
            "objective": None,
        })
        if existing is None:
            self.state["trials"].append(record)
        self.save()
        if self.args.dry_run:
            log_message(self.pipeline_log, f"Dry run: prepared {trial_id}.")
            return record

        console_log = self.console_dir / f"{trial_id}.log"
        command = [self.args.python, str(MAIN_SCRIPT), "--config", str(config_path)]
        environment = os.environ.copy()
        temp_dir = ROOT / ".tmp"
        temp_dir.mkdir(exist_ok=True)
        environment["TEMP"] = str(temp_dir)
        environment["TMP"] = str(temp_dir)
        log_message(
            self.pipeline_log,
            f"Start {trial_id}: phase={phase}, epochs={epochs}, stride={stride}, params={params}",
        )
        with console_log.open("w", encoding="utf-8") as console:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=console,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        metric = parse_metric(train_log, "Final Validation")
        record["returncode"] = process.returncode
        record["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        record["validation"] = metric
        if process.returncode == 0 and metric is not None:
            record["status"] = "completed"
            record["objective"] = validation_objective(
                metric,
                float(config["early_stop_fde_weight"]),
            )
            log_message(
                self.pipeline_log,
                f"Complete {trial_id}: ADE={metric['ade']:.5f}, FDE={metric['fde']:.5f}, "
                f"objective={record['objective']:.5f}.",
            )
        else:
            record["status"] = "failed"
            record["error"] = f"returncode={process.returncode}, final_validation={metric}"
            log_message(self.pipeline_log, f"Failed {trial_id}: {record['error']}.")
        self.save()
        return record

    def run_final_test(self, winner: dict) -> dict:
        source_config = json.loads(Path(winner["config"]).read_text(encoding="utf-8"))
        test_id = "best_once_test"
        test_config = dict(source_config)
        test_config.update({
            "eval_only": True,
            "checkpoint_path": winner["checkpoint"],
            "evaluate_test_each_epoch": False,
            "evaluate_final_test": True,
            "plot_count": int(self.base.get("plot_count", 16)),
            "results_dir": str(self.run_dir),
            "run_name": test_id,
            "append_log": False,
        })
        config_path = self.config_dir / "best_test_config.json"
        atomic_write_json(config_path, test_config)
        train_log = self.run_dir / test_id / "train.log"
        console_log = self.console_dir / f"{test_id}.log"
        command = [self.args.python, str(MAIN_SCRIPT), "--config", str(config_path)]
        environment = os.environ.copy()
        environment["TEMP"] = str(ROOT / ".tmp")
        environment["TMP"] = str(ROOT / ".tmp")
        log_message(self.pipeline_log, f"Start one-time final test for {winner['trial_id']}.")
        with console_log.open("w", encoding="utf-8") as console:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=console,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        metric = parse_metric(train_log, "Final Test")
        result = {
            "returncode": process.returncode,
            "config": str(config_path),
            "log": str(train_log),
            "metric": metric,
        }
        self.state["final_test"] = result
        self.save()
        log_message(self.pipeline_log, f"Final test finished: {metric}.")
        return result

    def write_summary(self, winner: dict) -> None:
        metric = winner["validation"]
        final_test = self.state.get("final_test", {}).get("metric")
        lines = [
            f"# {self.args.study_name}",
            "",
            "Selection rule: validation ADE + 0.2 * validation FDE. Test data was not used for ranking.",
            "",
            "## Best validation configuration",
            "",
            f"- Trial: `{winner['trial_id']}`",
            f"- Validation ADE: {metric['ade']:.5f} nmi ({metric['ade'] * 1852:.2f} m)",
            f"- Validation FDE: {metric['fde']:.5f} nmi ({metric['fde'] * 1852:.2f} m)",
            f"- Objective: {winner['objective']:.5f}",
            f"- Checkpoint: `{winner['checkpoint']}`",
            f"- Config: `{winner['config']}`",
            "",
            "## Tuned parameters",
            "",
        ]
        for key, value in winner["params"].items():
            lines.append(f"- `{key}`: `{value}`")
        if final_test:
            lines.extend([
                "",
                "## One-time final test",
                "",
                f"- ADE: {final_test['ade']:.5f} nmi ({final_test['ade'] * 1852:.2f} m)",
                f"- FDE: {final_test['fde']:.5f} nmi ({final_test['fde'] * 1852:.2f} m)",
            ])
        (self.root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def execute(self) -> None:
        if self.lock_path.exists() and not self.args.dry_run:
            try:
                owner_pid = int(self.lock_path.read_text(encoding="ascii").strip())
            except (OSError, ValueError):
                owner_pid = -1
            if process_is_running(owner_pid):
                raise RuntimeError(
                    f"Study lock belongs to running PID {owner_pid}: {self.lock_path}"
                )
            log_message(self.pipeline_log, f"Remove stale study lock for PID {owner_pid}.")
            self.lock_path.unlink()
        if not self.args.dry_run:
            self.lock_path.write_text(str(os.getpid()), encoding="ascii")
        try:
            candidates = screen_candidates(self.base)[:self.args.max_screen_trials]
            screen_records = []
            for index, candidate in enumerate(candidates, start=1):
                trial_id = f"screen_{index:02d}_{candidate['name']}"
                record = self.run_trial(
                    trial_id=trial_id,
                    phase="screen",
                    params=candidate["params"],
                    epochs=self.args.screen_epochs,
                    stride=self.args.screen_stride,
                    patience=self.args.screen_patience,
                    selector_warmup=self.args.screen_warmup,
                )
                if record.get("status") == "completed":
                    screen_records.append(record)
            if self.args.dry_run:
                log_message(self.pipeline_log, "Dry run complete; no training was started.")
                return
            if not screen_records:
                raise RuntimeError("All screening trials failed. See console logs.")
            screen_records.sort(key=lambda item: item["objective"])
            finalist_count = min(self.args.finalists, len(screen_records))
            final_records = []
            for rank, screen_record in enumerate(screen_records[:finalist_count], start=1):
                params = {
                    key: screen_record["params"][key]
                    for key in (
                        "learning_rate",
                        "lr_scheduler",
                        "lr_reduce_factor",
                        "lr_scheduler_patience",
                        "lr_min",
                        "subroute_residual_scale",
                        "balanced_intent_loss_weight",
                    )
                }
                trial_id = f"final_{rank:02d}_{screen_record['trial_id']}"
                record = self.run_trial(
                    trial_id=trial_id,
                    phase="final",
                    params=params,
                    epochs=self.args.final_epochs,
                    stride=1,
                    patience=self.args.final_patience,
                    selector_warmup=int(self.base.get("candidate_selector_warmup_epochs", 10)),
                )
                if record.get("status") == "completed":
                    final_records.append(record)
            if not final_records:
                raise RuntimeError("All full-resolution finalist trials failed.")
            final_records.sort(key=lambda item: item["objective"])
            winner = final_records[0]
            self.state["winner_trial_id"] = winner["trial_id"]
            self.state["winner_checkpoint"] = winner["checkpoint"]
            best_config_path = self.config_dir / "best_training_config.json"
            shutil.copyfile(winner["config"], best_config_path)
            self.state["best_training_config"] = str(best_config_path)
            self.save()
            log_message(
                self.pipeline_log,
                f"Winner: {winner['trial_id']} with objective {winner['objective']:.5f}.",
            )
            if self.args.run_final_test:
                self.run_final_test(winner)
            self.write_summary(winner)
        finally:
            if self.lock_path.exists() and not self.args.dry_run:
                self.lock_path.unlink()


def main() -> None:
    args = parse_args()
    if args.max_screen_trials < 1:
        raise ValueError("--max-screen-trials must be at least 1.")
    if args.finalists < 1:
        raise ValueError("--finalists must be at least 1.")
    Study(args).execute()


if __name__ == "__main__":
    main()
