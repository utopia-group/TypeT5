from spot.train import *


def accs_as_table_row(accs_dict: dict):
    def retrive(path: str):
        segs = path.split(".")
        target = accs_dict
        for s in segs:
            target = target[s]
        assert isinstance(target, CountedAcc)
        return f"{target.acc * 100:.2f}"

    row1 = {
        "full.all": "full_acc.full_acc",
        "calibrated.all": "acc.acc",
        "calibrated.simple": "acc.acc_by_simple.simple",
        "calibrated.complex": "acc.acc_by_simple.complex",
        "base.all": "base_acc.base_acc",
    }

    nums = [retrive(path) for path in row1.values()]
    print("Accuracies on all types:")
    print("header: ", list(row1.keys()))
    print(" & ".join(nums))

    row2 = {
        "full.all": "full_acc_common.full_acc_common",
        "calibrated.all": "acc_common.acc_common",
        "calibrated.simple": "acc_common.acc_common_by_simple.simple",
        "calibrated.complex": "acc_common.acc_common_by_simple.complex",
        "base.all": "base_acc_common.base_acc_common",
    }
    nums = [retrive(path) for path in row2.values()]
    print("Accuracies on common types:")
    print("header: ", list(row2.keys()))
    print(" & ".join(nums))


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
