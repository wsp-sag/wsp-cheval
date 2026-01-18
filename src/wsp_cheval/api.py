from __future__ import annotations

__all__ = [
    "ChoiceNode",
    "ExpressionGroup",
]

import abc
from collections import deque
from typing import TYPE_CHECKING, Dict, Generator, Hashable, List, Optional, Set, Union

import attr
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .exceptions import ModelNotReadyError
from .ldf import LinkedDataFrame
from .parsing.expr_items import ChainTuple
from .parsing.expressions import Expression
from .utils import convert_series, to_numpy

if TYPE_CHECKING:
    from .model import ChoiceModel

# region Tree


class ChoiceNode(object):
    def __init__(
        self,
        root: ChoiceModel,
        name: str,
        parent: Optional[ChoiceNode] = None,
        logsum_scale: float = 1.0,
        level: int = 0,
    ) -> None:
        if "." in name:
            raise ValueError('Choice node name cannot contain "."')
        if name == "_":
            raise ValueError('The name "_" by itself is reserved, but choice names can include it (.e.g. "choice_2")')
        if (logsum_scale <= 0.0) or (logsum_scale > 1.0):
            raise ValueError(f"Logsum scale must be in the interval (0, 1], got {logsum_scale}")

        self._root: ChoiceModel = root

        self._name = str(name)
        self._parent = parent
        self._logsum_scale = None
        self.logsum_scale = logsum_scale
        self._level = level
        self._children: Dict[str, ChoiceNode] = {}

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"ChoiceNode({self.name})"

    @property
    def logsum_scale(self) -> float:
        return self._logsum_scale

    @logsum_scale.setter
    def logsum_scale(self, value):
        if (value <= 0.0) or (value > 1.0):
            raise ValueError(f"Logsum scale must be in the interval (0, 1], got {value}")
        self._logsum_scale = float(value)

    @property
    def name(self) -> str:
        return self._name

    @property
    def parent(self) -> Optional[ChoiceNode]:
        return self._parent

    @property
    def level(self) -> int:
        return self._level

    @property
    def is_parent(self) -> bool:
        return len(self._children) > 0

    @property
    def n_children(self) -> int:
        return len(self._children)

    @property
    def full_name(self) -> str:
        ids = deque()
        node = self
        while node is not None:
            ids.appendleft(node.name)
            node = node.parent
        return ".".join(ids)

    def children(self) -> Generator[ChoiceNode, None, None]:
        yield from self._children.values()

    def max_level(self) -> int:
        max_level = self._level

        for c in self.children():
            max_level = max(max_level, c.max_level())

        return max_level

    def nested_id(self, max_level: int) -> tuple[str, ...]:
        retval = ["."] * max_level
        if self._parent is None:
            retval[0] = self._name
        else:
            cutoff = self._level + 1
            retval[:cutoff] = self._parent.nested_id(max_level)[:cutoff]
            retval[cutoff - 2] = self.name
        return tuple(retval)

    def add_choice(self, name: str, logsum_scale: float = 1.0) -> ChoiceNode:
        node = self._root._create_node(name, logsum_scale, parent=self)
        self._children[name] = node
        return node

    def clear(self) -> None:
        raise NotImplementedError()


# endregion


# region Expression containers


@attr.s
class ExpressionSubGroup:
    name: Hashable = attr.ib()
    simple_symbols: Set[str] = attr.ib(default=attr.Factory(set))
    chained_symbols: Set[str] = attr.ib(default=attr.Factory(set))
    expressions: List[Expression] = attr.ib(default=attr.Factory(list))

    def append(self, e: Expression) -> None:
        self.expressions.append(e)
        self.simple_symbols |= e.symbols
        for chain_name in e.chains.keys():
            self.chained_symbols.add(chain_name)

    def __add__(self, other: ExpressionSubGroup) -> ExpressionSubGroup:
        new = ExpressionSubGroup(self.name)
        for e in self.expressions:
            new.append(e)
        for e in other.expressions:
            new.append(e)
        return new

    def itersimple(self) -> Generator[str, None, None]:
        yield from self.simple_symbols

    def iterchained(self) -> Generator[str, None, None]:
        yield from self.chained_symbols

    def __iter__(self) -> Generator[Expression, None, None]:
        yield from self.expressions


class ExpressionGroup(object):
    def __init__(self) -> None:
        self._ungrouped_expressions: List[Expression] = []
        self._simple_symbols: Set[str] = set()
        self._chained_symbols: Set[str] = set()
        self._subgroups: Dict[Hashable, ExpressionSubGroup] = {}

    def append(self, e: str, group: Optional[Hashable] = None) -> None:
        # Parse the expression and look for invalid syntax and inconsistent usage. self._simple_symbols and
        # self._chained_symbols are modified in-place during parsing.
        expr = Expression.parse(e, self._simple_symbols, self._chained_symbols)
        if group is not None:
            if group not in self._subgroups:
                subgroup = ExpressionSubGroup(group)
                self._subgroups[group] = subgroup
            else:
                subgroup = self._subgroups[group]

            subgroup.append(expr)
        else:
            self._ungrouped_expressions.append(expr)
            self._simple_symbols |= expr.symbols
            for chain_name in expr.chains.keys():
                self._chained_symbols.add(chain_name)

    def clear(self) -> None:
        self._ungrouped_expressions.clear()
        self._subgroups.clear()
        self._simple_symbols.clear()
        self._chained_symbols.clear()

    def itersimple(self, *, groups: bool = True) -> Generator[str, None, None]:
        yield from self._simple_symbols
        if not groups:
            return
        for subgroup in self._subgroups.values():
            yield from subgroup.itersimple()

    def iterchained(self, *, groups: bool = True) -> Generator[str, None, None]:
        yield from self._chained_symbols
        if not groups:
            return
        for subgroup in self._subgroups.values():
            yield from subgroup.iterchained()

    def __iter__(self, *, groups: bool = True) -> Generator[Expression, None, None]:
        yield from self._ungrouped_expressions
        if not groups:
            return
        for subgroup in self._subgroups.values():
            yield from subgroup

    def __add__(self, other: ExpressionGroup) -> ExpressionGroup:
        new = ExpressionGroup()

        new._simple_symbols = self._simple_symbols | other._simple_symbols
        new._chained_symbols = self._chained_symbols | other._chained_symbols
        new._ungrouped_expressions = self._ungrouped_expressions + other._ungrouped_expressions

        new._subgroups = {}
        keys = set(self._subgroups.keys()) | set(other._subgroups.keys())
        for key in keys:
            key_in_self, key_in_other = key in self._subgroups, key in other._subgroups
            if key_in_self and key_in_other:
                new._subgroups[key] = self._subgroups[key] + other._subgroups[key]
            elif key_in_self:
                new._subgroups[key] = self._subgroups[key]
            elif key_in_other:
                new._subgroups[key] = other._subgroups[key]

        return new

    def tolist(self, raw: bool = True) -> Union[List[str], List[Expression]]:
        return [e.raw for e in self._ungrouped_expressions] if raw else [e for e in self._ungrouped_expressions]

    def get_group(self, name: Hashable) -> ExpressionSubGroup:
        return self._subgroups[name]

    def drop_group(self, name: Hashable) -> None:
        del self._subgroups[name]

    def copy(self) -> ExpressionGroup:
        new = ExpressionGroup()
        for e in self._ungrouped_expressions:
            new._ungrouped_expressions.append(e)
            new._simple_symbols |= e.symbols
            for chain_name in e.chains.keys():
                new._chained_symbols.add(chain_name)
        for name, subgroup in self._subgroups.items():
            new_sg = ExpressionSubGroup(name)
            for e in subgroup:
                new_sg.append(e)
            new._subgroups[name] = new_sg
        return new


# endregion


# region Symbols for scope


class AbstractSymbol(object, metaclass=abc.ABCMeta):
    def __init__(
        self,
        parent: ChoiceModel,
        name: str,
    ) -> None:
        self._parent = parent
        self._name = name

    @abc.abstractmethod
    def assign(self, data) -> None:
        pass

    @abc.abstractmethod
    def _get(self, **kwargs) -> Union[float, np.ndarray]:
        pass

    @abc.abstractmethod
    def empty(self) -> None:
        pass

    @abc.abstractmethod
    def copy(self, new_parent: ChoiceModel, copy_data: bool, row_mask: Optional[ArrayLike]) -> AbstractSymbol:
        pass

    @property
    @abc.abstractmethod
    def filled(self) -> bool:
        pass


class NumberSymbol(AbstractSymbol):
    def __init__(
        self,
        parent: ChoiceModel,
        name: str,
    ) -> None:
        super().__init__(parent, name)
        self._val = None

    def assign(self, data: float) -> None:
        self._val = float(data)

    def _get(self) -> float:
        if self._val is None:
            raise ModelNotReadyError()
        return self._val

    def empty(self) -> None:
        self._val = None

    def copy(self, new_parent: ChoiceModel, copy_data: bool, row_mask: Optional[ArrayLike] = None) -> NumberSymbol:
        new = NumberSymbol(new_parent, self._name)
        if copy_data:
            new._val = self._val

        return new

    @property
    def filled(self) -> bool:
        return self._val is not None


class VectorSymbol(AbstractSymbol):
    def __init__(
        self,
        parent: ChoiceModel,
        name: str,
        orientation: int,
    ) -> None:
        super().__init__(parent, name)

        if orientation not in {0, 1}:
            raise ValueError("Orientation must be 0 (column vector) or 1 (row vector)")
        self._orientation = orientation
        self._raw_array: Optional[np.ndarray] = None

    def assign(self, data: ArrayLike) -> None:
        index_to_check = self._parent.choices if self._orientation else self._parent.decision_units

        if isinstance(data, pd.Series):
            if not index_to_check.equals(data.index):
                raise ValueError("Series index does not match length of rows or columns")
            vector = convert_series(data, allow_raw=False)  # Convert Categorical/Text right away
        elif isinstance(data, np.ndarray):
            if len(data.shape) != 1:
                raise ValueError("Only 1D arrays are permitted")
            if len(data) != len(index_to_check):
                raise ValueError("Array length does not match length of rows or columns")
            vector = data
        else:
            raise TypeError(type(data))

        self._raw_array = vector[...]  # Shallow copy
        n = len(index_to_check)

        if self._orientation:
            self._raw_array.shape = 1, n
        else:
            self._raw_array.shape = n, 1

    def _get(self) -> np.ndarray:
        if self._raw_array is None:
            raise ModelNotReadyError
        return self._raw_array

    def empty(self) -> None:
        self._raw_array = None

    def copy(self, new_parent: ChoiceModel, copy_data: bool, row_mask: Optional[ArrayLike] = None) -> VectorSymbol:
        new = VectorSymbol(new_parent, self._name, self._orientation)
        if copy_data and self._raw_array is not None:
            if self._orientation == 0 and row_mask is not None:
                new_array = self._raw_array[row_mask]
            else:
                new_array = self._raw_array

            new._raw_array = new_array

        return new

    @property
    def filled(self) -> bool:
        return self._raw_array is not None


class TableSymbol(AbstractSymbol):
    def __init__(
        self,
        parent: ChoiceModel,
        name: str,
        orientation: int,
        mandatory_attributes: Optional[Set[str]] = None,
        allow_links: bool = True,
    ) -> None:
        super().__init__(parent, name)
        if orientation not in {0, 1}:
            raise ValueError("Orientation must be 0 or 1")
        self._orientation = orientation

        if mandatory_attributes is None:
            mandatory_attributes = set()
        self._mandatory_attributes = mandatory_attributes
        self._allow_links = bool(allow_links)
        self._table: Optional[pd.DataFrame] = None

    def assign(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        index_to_check = self._parent.decision_units if self._orientation == 0 else self._parent.choices
        if not data.index.equals(index_to_check):
            raise ValueError("DataFrame index does not match context rows or columns")

        for column in self._mandatory_attributes:
            if column not in data:
                raise ValueError(f"Mandatory attribute {column} not found in DataFrame")

        if not self._allow_links and not isinstance(data, LinkedDataFrame):
            raise TypeError(f"LinkedDataFrames not allowed for symbol {self._name}")

        self._table = data

    def _get(self, chain_info: Optional[ChainTuple] = None) -> np.ndarray:
        if chain_info is None:
            raise ValueError("chain_info cannot be None")

        chained = len(chain_info.chain) > 1

        if chained:
            if not isinstance(self._table, LinkedDataFrame):
                raise ValueError("Table must be a LinkedDataFrame for chained access")
            item = self._table
            for item_name in reversed(chain_info.chain):
                item = item[item_name]

            if chain_info.withfunc:
                series = getattr(item, chain_info.func)(chain_info.args)
            else:
                series = item
        else:
            attribute_name = chain_info.chain[0]
            series = self._table[attribute_name]

        vector = convert_series(series, allow_raw=False)

        n = len(vector)
        new_shape = (n, 1) if self._orientation == 0 else (1, n)
        vector.shape = new_shape
        return vector

    def empty(self) -> None:
        self._table = None

    def copy(self, new_parent: ChoiceModel, copy_data: bool, row_mask: Optional[ArrayLike] = None) -> TableSymbol:
        new = TableSymbol(new_parent, self._name, self._orientation, self._mandatory_attributes, self._allow_links)
        if copy_data and self._table is not None:
            if self._orientation == 0 and row_mask is not None:
                new._table = self._table.loc[row_mask]
            else:
                new._table = self._table
        return new

    @property
    def filled(self) -> bool:
        return self._table is not None


class MatrixSymbol(AbstractSymbol):
    def __init__(
        self,
        parent: ChoiceModel,
        name: str,
        orientation: int = 0,
        reindex_cols: bool = True,
        reindex_rows: bool = True,
        fill_value: float = 0.0,
    ) -> None:
        super().__init__(parent, name)
        self._matrix: Optional[np.ndarray] = None
        if orientation not in {0, 1}:
            raise ValueError("Orientation must be 0 or 1")
        self._orientation = orientation
        self._reindex_cols = reindex_cols
        self._reindex_rows = reindex_rows
        self._fill_value = fill_value

    def assign(self, data: pd.DataFrame) -> None:
        rows = self._parent.decision_units
        cols = self._parent.choices

        if self._orientation == 1:
            data = data.transpose()

        if isinstance(data, pd.DataFrame):
            rows_match = data.index is rows or rows.equals(data.index)
            cols_match = data.columns is cols or cols.equals(data.columns)

            if rows_match and cols_match:
                self._matrix = to_numpy(data)
            else:
                # Try to manually control the amount of excess RAM needed for partial utilities, as Pandas reindex()
                # is quite hungry. This is important to keep this feature scalable.

                matrix = np.full([len(rows), len(cols)], self._fill_value, dtype=to_numpy(data).dtype)
                if not cols_match:
                    assert self._reindex_cols
                    col_indexer = cols.get_indexer(data.columns)
                    if np.any(col_indexer < 0):
                        raise NotImplementedError("Cannot handle missing columns")
                    if len(col_indexer) != data.shape[1]:
                        raise ValueError(
                            f"Column indexer length {len(col_indexer)} does not match data columns {data.shape[1]}"
                        )
                else:
                    col_indexer = slice(None)

                if not rows_match:
                    assert self._reindex_rows
                    row_indexer = rows.get_indexer(data.index)
                    if np.any(row_indexer < 0):
                        raise NotImplementedError("Cannot handle missing rows")
                    if len(row_indexer) != data.shape[0]:
                        raise ValueError(
                            f"Row indexer length {len(row_indexer)} does not match data rows {data.shape[0]}"
                        )
                else:
                    row_indexer = slice(None)

                matrix[row_indexer, col_indexer] = data
                self._matrix = matrix
        else:
            raise TypeError(type(data))

    def _get(self) -> np.ndarray:
        return self._matrix

    def empty(self) -> None:
        self._matrix = None

    def copy(self, new_parent: ChoiceModel, copy_data: bool, row_mask: Optional[ArrayLike] = None) -> MatrixSymbol:
        new = MatrixSymbol(
            new_parent,
            self._name,
            self._orientation,
            self._reindex_cols,
            self._reindex_rows,
            fill_value=self._fill_value,
        )
        if copy_data and self._matrix is not None:
            if row_mask is not None:
                new._matrix = self._matrix[row_mask, :]
            else:
                new._matrix = self._matrix
        return new

    @property
    def filled(self) -> bool:
        return self._matrix is not None


# endregion
