from spot.train import *

TypeT5Default = TrainingConfig(
    func_only=True,
    pre_args=PreprocessArgs(
        drop_env_types=False,
        add_implicit_rel_imports=True,
    ),
    left_margin=2048,
    right_margin=2048 - 512,
    preamble_size=1000,
)


class AblationConfigs:
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
