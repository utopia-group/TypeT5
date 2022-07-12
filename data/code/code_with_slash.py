class SlashClass:
    def __init__(self, check_interval: int, folder: Path, /) -> None:
        self._autolocked: Dict[Path, int] = {}
        self._lockers: Dict[Path, "DirectEdit"] = {}
        self._to_lock: Items = []
