import random
from engine import Value

class Module:
		# typical training loop
		# zero_grad because grad is accumulated, so we need to reset it to 0 before each backward pass
		# 	model.zero_grad()              # reset old grads
		# 	out = model(x)                 # forward
		# 	loss = loss_fn(out, y)         # compute loss
		# 	loss.backward()                # populate p.grad for all params
		# 	for p in model.parameters():   # update (SGD example)
		#     p.data -= lr * p.grad
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

# neuron is a module, aka inherit from Module
# nin is the number of inputs
# nonlin is a boolean that determines if the neuron is nonlinear
class Neuron(Module):
	def __init__(self, nin, method='relu', nonlin=True, layer_idx=0):
		self.w = [Value(random.uniform(-1, 1), label=f'w{i}_{layer_idx}') for i in range(nin)]
		self.b = Value(0.0, label='b')
		self.nonlin = nonlin
		self.method = method
		# runtime caches for visualization
		self.last_x = None
		self.last_sum = None
		self.last_out = None

	# __call__ is a special method that allows an object to be called like a function
	def __call__(self, x):
		# zip is a built-in function that zips two lists together
		act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
		self.last_x = x
		self.last_sum = act
		out = act.tanh() if self.nonlin and self.method == 'tanh' else act.relu() if self.nonlin and self.method == 'relu' else act
		self.last_out = out
		return out

	def parameters(self):
		return self.w + [self.b]

	def __repr__(self):
		return f"{'Tanh' if self.nonlin and self.method == 'tanh' else 'ReLU' if self.nonlin and self.method == 'relu' else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):
	def __init__(self, nin, nout, method='relu', nonlin=True, layer_idx=0):
		self.neurons = [Neuron(nin, method, nonlin, layer_idx) for _ in range(nout)]

	def __call__(self, x):
		output = [n(x) for n in self.neurons]
		return output[0] if len(output) == 1 else output

	def parameters(self):
		return [p for neuron in self.neurons for p in neuron.parameters()]

	def __repr__(self):
		return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
	def __init__(self, nin, nouts, method='relu', nonlin=True):
		sizes = [nin] + nouts
		self.layers = [Layer(sizes[i], sizes[i+1], method, nonlin, layer_idx=i) for i in range(len(nouts))]
		self.last_inputs = None

	def __call__(self, x):
		self.last_inputs = x
		for layer in self.layers:
			x = layer(x)
		return x

	def parameters(self):
		return [p for layer in self.layers for p in layer.parameters()]

	def __repr__(self):
		return f"MLP of [{', '.join(str(l) for l in self.layers)}]"