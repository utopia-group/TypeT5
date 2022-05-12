def __init__(
    self, check_interval: int, folder: Path, /
) -> None:
    super().__init__(check_interval, "AutoLocker")

    self._autolocked: Dict[Path, int] = {}
    self._lockers: Dict[Path, "DirectEdit"] = {}
    self._to_lock: Items = []
