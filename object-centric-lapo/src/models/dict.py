from typing import Iterable, Mapping, Optional, Tuple, Union

import torch
from torch.nn import Module

Key = Union[str, Tuple[str, ...]]


class RemappingModuleDict(torch.nn.ModuleDict):
    """ModuleDict that allows using tuples as keys and converts dots to hashes.

    This class extends `torch.nn.ModuleDict` to support:
    - Keys containing dots (`.`) by internally converting them to hashes (`#`)
    - Tuple keys by converting them to a string representation
    - Keys that conflict with class attributes by wrapping them

    The internal representation is automatically converted back to the original format
    when accessing keys externally.

    Inspired by:
    https://github.com/pyg-team/pytorch_geometric/blob/5496e963f6638e032687a5b54bc7408ceec47639/torch_geometric/nn/module_dict.py

    Args:
        modules (Optional[Mapping[Union[str, Tuple[str, ...]], Module]]): A mapping of
            module names to modules. Keys can be strings (with dots) or tuples.

    Examples:
        >>> import torch.nn as nn
        >>> # Create a RemappingModuleDict with various key types
        >>> modules = RemappingModuleDict({
        ...     'encoder.layer1': nn.Linear(10, 20),
        ...     'encoder.layer2': nn.Linear(20, 30),
        ...     ('decoder', 'output'): nn.Linear(30, 10),
        ... })
        >>>
        >>> # Access modules using original keys
        >>> layer1 = modules['encoder.layer1']
        >>> output = modules[('decoder', 'output')]
        >>>
        >>> # Add new modules with dot notation
        >>> modules['model.head'] = nn.Linear(30, 5)
        >>>
        >>> # Check if keys exist
        >>> 'encoder.layer1' in modules  # True
        >>> ('decoder', 'output') in modules  # True
    """

    def __init__(
        self,
        modules: Optional[Mapping[Union[str, Tuple[str, ...]], Module]] = None,
    ):
        if modules is not None:  # Replace the keys in modules:
            modules = {self.to_internal_key(key): module for key, module in modules.items()}
        super().__init__(modules)

    @classmethod
    def to_internal_key(cls, key: Key) -> str:
        # ModuleDict cannot handle tuples as keys:
        if isinstance(key, tuple):
            assert len(key) > 1
            key = f"<{'___'.join(key)}>"
        assert isinstance(key, str)

        # ModuleDict cannot handle keys that exists as class attributes:
        if hasattr(cls, key):
            key = f"<{key}>"

        # ModuleDict cannot handle dots in keys:
        return key.replace(".", "#")

    @classmethod
    def to_external_key(cls, key: str) -> Key:
        key = key.replace("#", ".")

        if key[0] == "<" and key[-1] == ">" and hasattr(cls, key[1:-1]):
            key = key[1:-1]

        if key[0] == "<" and key[-1] == ">" and "___" in key:
            key = tuple(key[1:-1].split("___"))

        return key

    def __getitem__(self, key: Key) -> Module:
        return super().__getitem__(self.to_internal_key(key))

    def __setitem__(self, key: Key, module: Module):
        return super().__setitem__(self.to_internal_key(key), module)

    def __delitem__(self, key: Key):
        return super().__delitem__(self.to_internal_key(key))

    def __contains__(self, key: Key) -> bool:
        return super().__contains__(self.to_internal_key(key))

    def keys(self) -> Iterable[Key]:
        return [self.to_external_key(key) for key in super().keys()]

    def items(self) -> Iterable[Tuple[Key, Module]]:
        return [(self.to_external_key(k), v) for k, v in super().items()]
