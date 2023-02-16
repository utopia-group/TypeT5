from typet5.train import *


def accs_as_table_row(accs_dict: dict):
    def retrive(path: str):
        segs = path.split(".")
        target = accs_dict
        for s in segs:
            if s not in target:
                return "N/A"
            target = target[s]
        assert isinstance(target, CountedAcc), f"Unexpected type: {CountedAcc}"
        return f"{target.acc * 100:.2f}"

    def print_row(name: str, postfix: str):
        row = {
            "full.all": f"full_acc{postfix}.full_acc{postfix}",
            "calibrated.all": f"acc{postfix}.acc{postfix}",
            "calibrated.simple": f"acc{postfix}.acc{postfix}_by_simple.simple",
            "calibrated.complex": f"acc{postfix}.acc{postfix}_by_simple.complex",
            "base.all": f"base_acc{postfix}.base_acc{postfix}",
        }

        nums = [retrive(path) for path in row.values()]
        print(f"Accuracies on {name} types:")
        print("header: ", list(row.keys()))
        print(" & ".join(nums))

    print_row("all", "")
    print_row("common", "_common")
    print_row("rare", "_rare")


class TypeT5Configs:
    Default = TrainingConfig(
        func_only=True,
        pre_args=PreprocessArgs(
            drop_env_types=False,
            add_implicit_rel_imports=True,
        ),
        left_margin=2048,
        right_margin=2048 - 512,
        preamble_size=1000,
    )

    NoPreamble = TrainingConfig(
        func_only=True,
        pre_args=PreprocessArgs(
            imports_in_preamble=False,
            stub_in_preamble=False,
            drop_env_types=False,
            add_implicit_rel_imports=True,
        ),
        left_margin=2048,
        right_margin=2048 - 512,
        preamble_size=1000,
    )

    NoUsees = TrainingConfig(
        func_only=True,
        pre_args=PreprocessArgs(
            max_callees=0,
            drop_env_types=False,
            add_implicit_rel_imports=True,
        ),
        left_margin=512,
        right_margin=2048 + 1024,
        preamble_size=511,
    )

    NoUsers = TrainingConfig(
        func_only=True,
        pre_args=PreprocessArgs(
            max_callers=0,
            drop_env_types=False,
            add_implicit_rel_imports=True,
        ),
        left_margin=2048 + 1024,
        right_margin=512,
        preamble_size=1000,
    )

    NoSequential = TrainingConfig(
        func_only=True,
        pre_args=PreprocessArgs(
            drop_env_types=True,
            add_implicit_rel_imports=True,
        ),
        left_margin=2048,
        right_margin=2048 - 512,
        preamble_size=1000,
    )
