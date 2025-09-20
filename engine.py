import math

class Value:
		def __init__(self, data, _children=(), _op='', label=''):
			self.data = data
			self._prev = set(_children)
			self._op = _op
			self.label = label
			self.grad = 0.0

		def __repr__(self):
			return f"Value(data={self.data})"

		def __add__(self, other):
			out = Value(self.data + other.data, (self, other), '+')
			return out

		def __mul__(self, other):
			out = Value(self.data * other.data, (self, other), '*')
			return out

		def tanh(self):
			x = self.data
			t = (math.exp(2* x) - 1) / (math.exp(2* x) + 1)
			out = Value(t, (self,), 'tanh')
			return out