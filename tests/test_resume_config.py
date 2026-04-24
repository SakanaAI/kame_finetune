import argparse
import json
from pathlib import Path

import finetune


def _build_args(
    tmp_path,
    monkeypatch,
    *,
    report_to=None,
    include_run_id=False,
):
    output_dir = tmp_path / "output"
    checkpoint_dir = output_dir / "step_2"
    checkpoint_dir.mkdir(parents=True)

    ds_config = (
        Path(__file__).resolve().parents[1] / "ds_configs" / "zero3-bfp16-warmlr-act_ckpt.json"
    )
    monkeypatch.setenv("ACCELERATE_USE_DEEPSPEED", "true")
    monkeypatch.setenv("ACCELERATE_DEEPSPEED_CONFIG_FILE", str(ds_config))

    parser = argparse.ArgumentParser()
    finetune.setup_argparser(parser)
    cli_args = [
        "--output_dir",
        str(output_dir),
        "--train_data_files",
        "processed_data/sample.parquet",
        "--model_dir",
        "init_models/sample",
        "--resume_from_checkpoint",
        str(checkpoint_dir),
    ]
    if report_to is not None:
        cli_args += ["--report_to", report_to]
    args = parser.parse_args(cli_args)

    prev_config = vars(args).copy()
    prev_config["use_deepspeed"] = True
    prev_config["deepspeed_config_file"] = str(ds_config)
    prev_config["with_tracking"] = report_to is not None
    if include_run_id:
        prev_config["run_id"] = "wandb-run-id"

    (output_dir / "config.json").write_text(json.dumps(prev_config), encoding="utf-8")
    return args


def test_postprocess_args_allows_resume_without_tracking_run_id(tmp_path, monkeypatch):
    args = _build_args(tmp_path, monkeypatch, report_to=None, include_run_id=False)

    finetune.postprocess_args(args)

    assert args.with_tracking is False
    assert args.run_id_to_resume is None


def test_postprocess_args_restores_run_id_when_tracking_is_enabled(tmp_path, monkeypatch):
    args = _build_args(tmp_path, monkeypatch, report_to="wandb", include_run_id=True)

    finetune.postprocess_args(args)

    assert args.with_tracking is True
    assert args.run_id_to_resume == "wandb-run-id"
