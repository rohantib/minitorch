from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    args_forward = (x + epsilon if i == arg else x for i, x in enumerate(vals))
    args_backward = (x - epsilon if i == arg else x for i, x in enumerate(vals))
    return (f(*args_forward) - f(*args_backward)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    visited = set()
    top_rev_order: List[Variable] = []

    def top_helper(variable: Variable) -> None:
        if variable.unique_id in visited or variable.is_constant():
            return
        visited.add(variable.unique_id)
        for var in variable.parents:
            top_helper(var)
        top_rev_order.append(variable)

    top_helper(variable)
    return reversed(top_rev_order)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # NOTE: This safety check might not be necessary
    if deriv is None:
        deriv = 1.0
    deriv = float(
        deriv
    )  # In case an integer input was passed in, don't want to risk propagating it
    top_order = topological_sort(variable)
    intermediate_derivs = {variable.unique_id: deriv}
    for var in top_order:
        if var.is_leaf():
            continue
        d = intermediate_derivs.pop(var.unique_id)
        for parent, chain_d in var.chain_rule(d):
            if parent.is_leaf():
                parent.accumulate_derivative(chain_d)
            else:
                if parent.unique_id not in intermediate_derivs:
                    intermediate_derivs[parent.unique_id] = 0.0
                intermediate_derivs[parent.unique_id] += chain_d


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
