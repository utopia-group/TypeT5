def read_file(path: str) -> str:
    """read file content as string."""
    with open(path, "r") as f:
        return f.read()

