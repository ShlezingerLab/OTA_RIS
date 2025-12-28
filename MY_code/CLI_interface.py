import os
import subprocess
import sys


def _run(script_name: str, script_args: list[str]) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(here, script_name)
    cmd = [sys.executable, script_path, *script_args]
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    if not argv or argv[0] in {"-h", "--help"}:
        print(
            "usage: playground.py {train,test} [args...]\n\n"
            "Single entrypoint to run training or testing with your chosen args.\n\n"
            "Examples:\n"
            "  python MY_code/playground.py train --epochs 5 --batchsize 64\n"
            "  python MY_code/playground.py test --checkpoint MY_code/models_dict/minn_model.pth\n"
            "  python MY_code/playground.py train -h   (shows training.py help)\n"
            "  python MY_code/playground.py test -h    (shows test.py help)\n"
        )
        return 0

    command = argv[0]
    forwarded = argv[1:]

    if command == "train":
        return _run("training.py", forwarded)
    if command == "test":
        return _run("test.py", forwarded)

    print(f"[ERROR] Unknown command: {command}\n")
    print("Run: python MY_code/playground.py -h")
    return 2


if __name__ == "__main__":
    IDE_COMMAND = "train"
    IDE_TRAIN_STAGE = 5  # 0: CNN, 1: Encoder, 2: Controller, 3: Decoder, 4: E2E, 5: Debug (Enc+Dec)

    IDE_TRAIN_ARGS: dict[str, object] = {
        "--N_t": 8,    # Nt
        "--N_r": 12,   # Nr
        "--N_m": 64,
        "--subset_size": 1000,          # 60000
        "--batchsize": 100,            # 256
        "--channel_sampling_size": 100,  # 10000
        "--lr": 1e-3,
        "--weight_decay": 1e-7,
        "--metasurface_type": "ris",
        "--combine_mode": "metanet",
        "--cotrl_CSI": True,
        "--channel_type": "geometric_ricean",
        "--noise_std":  1e-6,
        "--tx_power_dbm": 30.0, # 30 dBm = 1 W
        "--geo_pathloss_exp": 2.0,
        "--geo_pathloss_gain_db": 70.0, # TODO: It has to be ~60
        "--k_factor_db": 3.0,  # direct-link K-factor in dB (H1/H2 use 13/7 inside training.py)
        "--epochs": 150,
    }

    # Select the stage configuration based on IDE_TRAIN_STAGE
    STAGED_CONFIGS = {
        0: { # PHASE 0: Train CNN Classifier (Teacher)
            "--train_classifier": True,
            # #"--teacher_use_channel": True,
            # #"--teacher_channel_output_mode": "magnitude",
            # #"--teacher_channel_noise_std": 1e-2,
            "--classifier_path": "models_dict/cnn_classifier_teacher.pth",
            "--plot_path": "plots/phase0_cnn.png",
        },
        1: { # PHASE 1: Train Encoder via Distillation (Stage 2)
            "--stage": 2,
            "--teacher_path": "models_dict/cnn_classifier_teacher.pth",
            "--save_path": "models_dict/phase1_encoder.pth",
            "--plot_path": "plots/phase1_encoder.png",
        },
        2: { # PHASE 2: Train Controller via Distillation (Stage 3)
            "--stage": 3,
            "--teacher_path": "models_dict/cnn_classifier_teacher.pth",
            "--load_path": "models_dict/phase1_encoder.pth",
            "--save_path": "models_dict/phase2_ctrl.pth",
            "--grad_approx": True, # Toggle here to use gradient approximation
            "--grad_approx_sigma": 0.1,
            "--plot_path": "plots/phase2_ctrl_app.png",
        },
        3: { # PHASE 3: Train Decoder (Stage 4)
            "--stage": 4,
            "--load_path": "models_dict/phase2_ctrl.pth",
            "--save_path": "models_dict/phase3_decoder.pth",
            "--plot_path": "plots/phase3_decoder.png",
        },
        5: { # NEW: Phase 4: Debug mode - Train Encoder and Decoder (Controller frozen)
            "--stage": 5,
            "--load_path": "models_dict/phase2_ctrl.pth",
            "--save_path": "models_dict/phase4_debug.pth",
            "--plot_path": "plots/phase4_debug.png",
        },
        4: { # Default / End-to-End training
            "--encoder_distill": False,
            "--plot_path": "plots/E2E.png",
        }
    }

    if IDE_TRAIN_STAGE in STAGED_CONFIGS:
        IDE_TRAIN_ARGS.update(STAGED_CONFIGS[IDE_TRAIN_STAGE])
######################################################################
    IDE_TEST_ARGS: dict[str, object] = {
        "--compare_checkpoints": (
            "minn_model_teacher_encoder_distill=False.pth",
            "minn_model_student_encoder_distill=True.pth",
        ),
        #"--checkpoint": "minn_model_teacher_fading_type=rayleigh_meta.pth",
        "--num_trials": 10,
        "--subset_size": 1000,
        "--batchsize": 100,
        "--channel_sampling_size": 100,
        "--N_t": 10,  # or [10, 15]
        "--N_r": 8,
        "--N_m": 9,
        "--combine_mode": "direct",
        "--cotrl_CSI": True,
        "--noise_std": 1,
        "--channel_type": "geometric_ricean",
        "--k_factor_db": 3.0,
        "--plot_path": "plots/test_summary_1.png",
    }

    def _safe_token(s: str) -> str:
        # Make a filename-safe token (good enough for Windows paths).
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        return "".join((ch if ch in allowed else "_") for ch in s)

    def _suffix_path(path: str, suffix: str) -> str:
        root, ext = os.path.splitext(path)
        return f"{root}{suffix}{ext}"

    def _suffix_save_path(base_path: str, sweep_key: str, sweep_val: object) -> str:
        """
        Suffix save_path based on the swept key/value.

        User desired example (POSIX): training_N_r:10.pth
        On Windows, ':' is not allowed in filenames, so we use '=' instead.
        """
        key = sweep_key.lstrip("-")
        sep = ":" if os.name != "nt" else "="
        tag = f"{_safe_token(key)}{sep}{_safe_token(str(sweep_val))}"
        root, ext = os.path.splitext(base_path)
        if root.endswith("_"):
            return f"{root}{tag}{ext}"
        return f"{root}_{tag}{ext}"

    def _is_dir_like_path(p: str) -> bool:
        # Treat trailing path separator as "directory intent" (works even if path doesn't exist yet).
        if not p:
            return False
        return p.endswith(("/", "\\"))

    def _is_abs_path(p: str) -> bool:
        try:
            return os.path.isabs(p)
        except Exception:
            return False

    def _prefix_models_dict(p: str) -> str:
        """
        For convenience in IDE runs: treat checkpoint paths as relative to MY_code/models_dict/
        unless they are absolute or already start with that prefix.
        """
        if not p:
            return p
        if not os.path.isabs(p):
            # Prepend the script's directory for relative paths
            here = os.path.dirname(os.path.abspath(__file__))
            return os.path.join(here, "models_dict", p)
        return p

    DEFAULT_CNN_CLASSIFIER_PATH = "models_dict/cnn_classifier.pth"
    DEFAULT_STUDENT_STORE_NAME = "minn_model_student"
    DEFAULT_BASE_MODEL_STORE_NAME = "minn_model"

    def _default_teacher_path_for_fd(arg_dict: dict[str, object]) -> str:
        # Only meaningful when encoder_distill is enabled; allow overriding via IDE_TRAIN_ARGS.
        # Returns path to CNN classifier to use as teacher.
        v = arg_dict.get("--teacher_path")
        return str(v) if isinstance(v, str) and v else DEFAULT_CNN_CLASSIFIER_PATH

    def _model_store_name_for_save(arg_dict: dict[str, object]) -> str:
        # If staged training is enabled, use stage name
        stage = arg_dict.get("--stage")
        if stage is not None:
            return f"minn_model_stage{stage}"

        # Save student model when distilling; save base model when not distilling.
        v = arg_dict.get("--encoder_distill")
        # If it's a list, we are in "compare" intent; default to student naming so users don't overwrite base by accident.
        if isinstance(v, list):
            return DEFAULT_STUDENT_STORE_NAME
        return DEFAULT_STUDENT_STORE_NAME if bool(v) else DEFAULT_BASE_MODEL_STORE_NAME

    def _resolve_train_save_path(
        save_path_value: str,
        arg_dict: dict[str, object],
        sweep_key: str | None,
        sweep_val: object | None,
    ) -> str:
        """
        If save_path is a directory, save under:
          {save_dir}/{minn_model|minn_model_student}.pth
        Otherwise, treat it as a file path and (optionally) suffix it.
        """
        if save_path_value == "":
            return ""

        if not os.path.isabs(save_path_value):
            # Prepend the script's directory for relative paths
            here = os.path.dirname(os.path.abspath(__file__))
            save_path_value = os.path.join(here, save_path_value)

        if _is_dir_like_path(save_path_value):
            store_name = _model_store_name_for_save(arg_dict)
            resolved = os.path.join(save_path_value, f"{store_name}.pth")
        else:
            resolved = save_path_value

        if sweep_key is not None:
            resolved = _suffix_save_path(resolved, sweep_key, sweep_val)
        return resolved

    def _resolve_train_plot_path(
        plot_path_value: str,
        arg_dict: dict[str, object],
        sweep_key: str | None,
        sweep_val: object | None,
    ) -> str:
        """
        Resolve plot_path into a full path, and apply sweep suffixing when needed.
        """
        if plot_path_value == "":
            return ""

        if not os.path.isabs(plot_path_value):
            # Prepend the script's directory for relative paths
            here = os.path.dirname(os.path.abspath(__file__))
            plot_path_value = os.path.join(here, plot_path_value)

        if _is_dir_like_path(plot_path_value):
            # Directory intent: use a default filename.
            resolved = os.path.join(plot_path_value, "training_curves.png")
        else:
            resolved = plot_path_value

        if sweep_key is not None:
            resolved = _suffix_save_path(resolved, sweep_key, sweep_val)
        return resolved

    def _resolve_plain_plot_path(
        plot_path_value: str,
        sweep_key: str | None,
        sweep_val: object | None,
        default_filename: str,
    ) -> str:
        """
        Resolve plot_path without adding subdirs; used for IDE_TEST_ARGS.
        Only adds a filename if plot_path is directory-like, and applies sweep suffixing.
        """
        if plot_path_value == "":
            return ""

        if not os.path.isabs(plot_path_value):
            # Prepend the script's directory for relative paths
            here = os.path.dirname(os.path.abspath(__file__))
            plot_path_value = os.path.join(here, plot_path_value)

        resolved = plot_path_value
        if _is_dir_like_path(resolved):
            resolved = os.path.join(resolved, default_filename)
        if sweep_key is not None:
            resolved = _suffix_save_path(resolved, sweep_key, sweep_val)
        return resolved

    def _resolve_generic_path(p: str) -> str:
        if not p or os.path.isabs(p):
            return p
        # Prepend the script's directory for relative paths
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(here, p)

    def _build_cli_runs(
        arg_dict: dict[str, object],
        disable_default_plot_path: bool = False,
        disable_default_save_path: bool = False,
        plot_implies_show: bool = False,
        plot_role_subdir: bool = False,
        default_plot_filename: str = "test_summary.png",
        checkpoint_is_under_models_dict: bool = False,
        collapse_sweep_to_compare_arg: bool = False,
    ) -> list[list[str]]:
        def _effective_encoder_distill_for_run(
            *,
            sweep_key: str | None,
            sweep_v: object | None,
            compare_key: str | None,
            compare_vals: list[object] | None,
        ) -> bool:
            """
            Determine whether encoder distillation should be considered enabled for teacher_path injection.
            """
            # Check for stages 2 and 3 (Distillation stages)
            stage_val = arg_dict.get("--stage")
            if isinstance(stage_val, list):
                if any(v in [2, 3] for v in stage_val): return True
            elif stage_val in [2, 3]:
                return True

            raw = arg_dict.get("--encoder_distill")
            if isinstance(raw, list):
                # Collapsed compare run: treat as enabled if ANY compared value is truthy.
                if compare_key == "encoder_distill" and compare_vals is not None:
                    return any(bool(x) for x in compare_vals)
                # Sweep run: per-run decision.
                if sweep_key == "--encoder_distill":
                    return bool(sweep_v)
                return any(bool(x) for x in raw)
            return bool(raw)

        sweep_key: str | None = None
        sweep_vals: list[object] | None = None
        for k, v in arg_dict.items():
            if isinstance(v, list):
                if sweep_key is not None:
                    raise ValueError(
                        f"Only one arg may be a list (sweep). Found: {sweep_key} and {k}. "
                        f"Use tuples for multi-value flags."
                    )
                sweep_key = k
                sweep_vals = v

        if sweep_key is None:
            sweep_vals = [None]

        runs: list[list[str]] = []
        # For IDE test runs: if a list is provided, collapse into a single run with --compare_arg.
        compare_key: str | None = None
        compare_vals: list[object] | None = None
        if collapse_sweep_to_compare_arg and (sweep_key is not None) and (sweep_vals is not None):
            compare_key = sweep_key.lstrip("-")
            compare_vals = list(sweep_vals)
            sweep_vals = [None]

        for sweep_v in sweep_vals or [None]:
            # If we're doing a true sweep (multiple runs), make a per-run view of the args dict
            # so helper logic doesn't see the raw list value (e.g. bool([True, False]) == True).
            arg_dict_run = arg_dict
            if (compare_key is None) and (sweep_key is not None) and isinstance(arg_dict.get(sweep_key), list):
                arg_dict_run = dict(arg_dict)
                arg_dict_run[sweep_key] = sweep_v

            args: list[str] = []
            for k, v in arg_dict.items():
                if v is None:
                    continue
                if isinstance(v, list):
                    # If we collapsed sweep into compare_arg, skip emitting this key/value.
                    if compare_key is not None:
                        continue
                    v = sweep_v
                # Friendly aliases (so you can write "plot_acc" rather than "no_plot_acc").
                if k == "--plot_acc":
                    # training.py expects --no_plot_acc; so plot_acc=False => add --no_plot_acc
                    if isinstance(v, bool) and (not v):
                        args.append("--no_plot_acc")
                    continue
                # show_plot_end is always true in this playground: never pass '--no_show_plot_end'
                if k in {"--show_plot_end", "--no_show_plot_end"}:
                    continue

                if isinstance(v, bool):
                    if v:
                        args.append(k)
                    continue
                if isinstance(v, tuple):
                    args.append(k)
                    args += [str(x) for x in v]
                    continue
                args += [k, str(v)]

            if compare_key is not None and compare_vals is not None:
                args += ["--compare_arg", compare_key, *[str(x) for x in compare_vals]]

            # Teacher-path is only relevant for feature distillation.
            if _effective_encoder_distill_for_run(
                sweep_key=sweep_key,
                sweep_v=sweep_v,
                compare_key=compare_key,
                compare_vals=compare_vals,
            ):
                if "--teacher_path" not in arg_dict:
                    args += ["--teacher_path", _resolve_generic_path(_default_teacher_path_for_fd(arg_dict_run))]
            else:
                # Ensure we do not pass teacher_path when not distilling (treat as None).
                # (If it was accidentally present in the dict, remove it from the args list.)
                if "--teacher_path" in args:
                    try:
                        i = args.index("--teacher_path")
                        del args[i:i+2]
                    except Exception:
                        pass

            # Resolve other common path arguments to absolute paths
            for path_arg in ["--teacher_path", "--load_path", "--classifier_path"]:
                if path_arg in args:
                    try:
                        idx = args.index(path_arg)
                        args[idx + 1] = _resolve_generic_path(args[idx + 1])
                    except Exception:
                        pass

            # For IDE test runs: checkpoint paths are assumed to be under MY_code/models_dict/ by default.
            if checkpoint_is_under_models_dict and ("--checkpoint" in arg_dict) and isinstance(arg_dict.get("--checkpoint"), str):
                try:
                    i = args.index("--checkpoint")
                    args[i + 1] = _prefix_models_dict(str(args[i + 1]))
                except Exception:
                    pass

            # If plot_path wasn't provided, explicitly disable saving by overriding
            # training.py's default plot_path with an empty string.
            if disable_default_plot_path and ("--plot_path" not in arg_dict):
                args += ["--plot_path", ""]

            # If save_path wasn't provided, explicitly disable saving by overriding
            # training.py's default save_path behavior with an empty string.
            if disable_default_save_path and ("--save_path" not in arg_dict):
                args += ["--save_path", ""]

            # For IDE test runs: if plot_path is provided, default to enabling plotting.
            if plot_implies_show and ("--plot" not in arg_dict) and ("--plot_path" in arg_dict):
                plot_path_val = arg_dict.get("--plot_path")
                if isinstance(plot_path_val, str) and plot_path_val != "":
                    args.append("--plot")

            # For IDE test runs: if --plot is enabled, default to also showing the plot.
            if plot_implies_show and (("--plot" in arg_dict and bool(arg_dict.get("--plot"))) or ("--plot" in args)) and ("--plot_show" not in arg_dict):
                args.append("--plot_show")

            # Auto-suffix save_path to prevent overwrites when sweeping.
            if sweep_key is not None:
                if "--save_path" in arg_dict and isinstance(arg_dict.get("--save_path"), str):
                    try:
                        i = args.index("--save_path")
                        base = args[i + 1]
                        # If we collapsed into --compare_arg, don't mutate save_path here; training.py
                        # will save per compared value with its own suffixing.
                        if compare_key is None:
                            args[i + 1] = _resolve_train_save_path(base, arg_dict_run, sweep_key, sweep_v)
                    except Exception:
                        pass
                # Auto-suffix plot_path to prevent overwrites when sweeping.
                if "--plot_path" in arg_dict and isinstance(arg_dict.get("--plot_path"), str):
                    try:
                        i = args.index("--plot_path")
                        base = args[i + 1]
                        # If we collapsed into --compare_arg, suffix the plot ONCE with the compared key
                        # (avoid the previous behavior of suffixing with "<key>=None").
                        if compare_key is not None:
                            if plot_role_subdir:
                                resolved = _resolve_train_plot_path(base, arg_dict, None, None)
                            else:
                                resolved = _resolve_plain_plot_path(base, None, None, default_plot_filename)
                            args[i + 1] = _suffix_path(resolved, f"_{compare_key}")
                        else:
                            if plot_role_subdir:
                                args[i + 1] = _resolve_train_plot_path(base, arg_dict_run, sweep_key, sweep_v)
                            else:
                                args[i + 1] = _resolve_plain_plot_path(base, sweep_key, sweep_v, default_plot_filename)
                    except Exception:
                        pass
            else:
                # No sweep: still resolve directory-style save_path into default filename.
                if "--save_path" in arg_dict and isinstance(arg_dict.get("--save_path"), str):
                    try:
                        i = args.index("--save_path")
                        base = args[i + 1]
                        args[i + 1] = _resolve_train_save_path(base, arg_dict_run, None, None)
                    except Exception:
                        pass
            # No sweep: resolve plot_path (train: support dir-like values; test: support dir-like values).
            if "--plot_path" in arg_dict and isinstance(arg_dict.get("--plot_path"), str):
                try:
                    i = args.index("--plot_path")
                    base = args[i + 1]
                    if plot_role_subdir:
                        args[i + 1] = _resolve_train_plot_path(base, arg_dict_run, None, None)
                    else:
                        args[i + 1] = _resolve_plain_plot_path(base, None, None, default_plot_filename)
                except Exception:
                    pass

            runs.append(args)
        return runs

    argv = sys.argv[1:]
    if not argv:
        config = IDE_TRAIN_ARGS if IDE_COMMAND == "train" else IDE_TEST_ARGS

        # Special case: classifier training mode - use simple arg passing
        # Check if --train_classifier is True (not a list for comparison)
        train_classifier_val = config.get("--train_classifier")
        if train_classifier_val is True or (isinstance(train_classifier_val, list) and any(train_classifier_val)):
            if isinstance(train_classifier_val, list):
                print("[WARNING] --train_classifier should be True (not a list). Using first truthy value.")
                # For lists, we'll just run with the first truthy value
                config = dict(config)
                config["--train_classifier"] = next((v for v in train_classifier_val if v), True)

            simple_args = []
            for k, v in config.items():
                if v is None or v is False:
                    continue
                if k == "--train_classifier" and v is True:
                    simple_args.append("--train_classifier")
                    continue
                if isinstance(v, bool):
                    if v:
                        simple_args.append(k)
                    continue
                if isinstance(v, tuple):
                    simple_args.append(k)
                    simple_args += [str(x) for x in v]
                    continue
                if isinstance(v, list):
                    # Skip lists in classifier mode (not supported)
                    continue
                simple_args += [k, str(v)]
            rc = main([IDE_COMMAND, *simple_args])
            raise SystemExit(rc)

        rc = 0
        for run_args in _build_cli_runs(
            config,
            disable_default_plot_path=(IDE_COMMAND == "train"),
            disable_default_save_path=(IDE_COMMAND == "train"),
            plot_implies_show=(IDE_COMMAND == "test"),
            plot_role_subdir=(IDE_COMMAND == "train"),
            default_plot_filename=("training_curves.png" if IDE_COMMAND == "train" else "test_summary.png"),
            checkpoint_is_under_models_dict=(IDE_COMMAND == "test"),
            collapse_sweep_to_compare_arg=(IDE_COMMAND in {"test", "train"}),
        ):
            rc = main([IDE_COMMAND, *run_args])
            if rc != 0:
                break
        raise SystemExit(rc)

    raise SystemExit(main(argv))
