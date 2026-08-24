class NamedArray:
    def __new__(cls, *args, **kwargs):
        if cls is NamedArray:
            raise TypeError(
                "NamedLattice cannot be instantiated directly. Use `lynqx.named` creation functions instead."
            )
        return super().__new__(cls, *args, **kwargs)
