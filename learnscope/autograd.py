"""A tiny scalar autograd engine (the idea and some naming follow Andrej Karpathy's "micrograd").

Every operation (+, *, tanh ...) creates a NEW Value that remembers its parents and a small
closure holding the LOCAL derivative of that operation. `backward()` applies the chain rule
through the whole graph. This powers the Backprop explainer, so every number shown there is
computed live by this code.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable


class Value:
    __slots__ = ("data", "grad", "op", "prev", "label", "_backward")

    def __init__(self, data: float, prev: Iterable["Value"] = (), op: str = "", label: str = "") -> None:
        self.data = float(data)
        self.grad = 0.0
        self.op = op
        self.prev = tuple(prev)
        self.label = label
        self._backward: Callable[[], None] = lambda: None

    # ------------------------------------------------------------------ operations
    @staticmethod
    def of(x: "float | Value") -> "Value":
        return x if isinstance(x, Value) else Value(x)

    def add(self, other: "float | Value") -> "Value":
        """out = a + b.   d(out)/da = 1,  d(out)/db = 1"""
        o = Value.of(other)
        out = Value(self.data + o.data, (self, o), "+")

        def _backward() -> None:
            self.grad += out.grad
            o.grad += out.grad

        out._backward = _backward
        return out

    def mul(self, other: "float | Value") -> "Value":
        """out = a * b.   d(out)/da = b,  d(out)/db = a"""
        o = Value.of(other)
        out = Value(self.data * o.data, (self, o), "*")

        def _backward() -> None:
            self.grad += o.data * out.grad
            o.grad += self.data * out.grad

        out._backward = _backward
        return out

    def pow(self, k: float) -> "Value":
        """out = a ** k for a constant k.   d(out)/da = k * a**(k-1)"""
        out = Value(self.data**k, (self,), f"**{k}")

        def _backward() -> None:
            self.grad += k * self.data ** (k - 1) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        """out = tanh(a).   d(out)/da = 1 - tanh(a)^2 = 1 - out^2"""
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        """out = max(0, a).   d(out)/da = 1 if a > 0 else 0"""
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward() -> None:
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> "Value":
        """out = 1 / (1 + e^-a).   d(out)/da = out * (1 - out)"""
        s = 1.0 / (1.0 + math.exp(-self.data)) if self.data > -700 else 0.0
        out = Value(s, (self,), "sigmoid")

        def _backward() -> None:
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Value":
        """out = e^a.   d(out)/da = e^a = out"""
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward() -> None:
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Value":
        """out = ln(a).   d(out)/da = 1 / a"""
        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def neg(self) -> "Value":
        return self.mul(-1.0)

    def sub(self, other: "float | Value") -> "Value":
        return self.add(Value.of(other).neg())

    def div(self, other: "float | Value") -> "Value":
        return self.mul(Value.of(other).pow(-1.0))

    # python operators, for readable math
    __add__ = add
    __mul__ = mul
    __sub__ = sub
    __truediv__ = div
    __neg__ = neg
    __pow__ = pow

    def __radd__(self, other: float) -> "Value":
        return self.add(other)

    def __rmul__(self, other: float) -> "Value":
        return self.mul(other)

    def __rsub__(self, other: float) -> "Value":
        return Value.of(other).sub(self)

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"

    # ------------------------------------------------------------------ backprop
    def topo(self) -> list["Value"]:
        """Nodes in dependency order: every node appears AFTER all of its parents.
        Iterative so very deep graphs cannot overflow the recursion limit."""
        order: list[Value] = []
        seen: set[int] = set()
        stack: list[tuple[Value, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for parent in node.prev:
                if id(parent) not in seen:
                    stack.append((parent, False))
        return order

    def backward(self) -> None:
        """Set d(this)/d(this) = 1, then apply the chain rule in reverse topological order."""
        order = self.topo()
        for v in order:
            v.grad = 0.0
        self.grad = 1.0
        for v in reversed(order):
            v._backward()


def v(x: float, label: str = "") -> Value:
    return Value(x, (), "", label)
