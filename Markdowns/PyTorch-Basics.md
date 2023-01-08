### Tensor
In PyToch, data is represented as tensors, which are multi-dimensional arrays with a uniform type. You can create a tensor from a Python list or numpy array using the torch.Tensor() function
```python
import torch

# Create a 2D tensor with shape (3, 4)
tensor = torch.Tensor([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(tensor)
```

You can also create tensors with specific shapes and initial values using functions like torch.zeros(), torch.ones(), and torch.eye().
```Python
# Create a 2D tensor with shape (3, 4) filled with zeros
tensor = torch.zeros(3, 4)
print(tensor)

# Create a 2D tensor with shape (3, 4) filled with ones.
tensor = torch.ones(3, 4)
print(tensor)

# Create a 2D tensor with shape (3, 3) with the identity matrix
tensor = torch.eye(3)
print(tensor)
```

### Operations
You can perform various operations on tensors, such as element-wise arithmetic and matrix multiplication. For example:
```Python
import torch

# Create two tensors
tensor1 = torch.Tensor([[1,2,3], [4,5,6]])
tensor2 = torch.Tensor([[7,8,9], [10,11,12]])

# Element-wise addition
tensor3 = tensor1 + tensor2
print(tensor3)

# Element-wise substraction
tensor3 = tensor1 - tensor2
print(tensor3)

# Element-wise multiplication
tensor3 = tensor1 * tensor2
print(tensor3)

# Matrix multiplication
tensor3 = tensor1.mm(tensor2.t())
print(tensor3)
```
Element-wise vs matrix multiplication 
http://yetanothermathprogrammingconsultant.blogspot.com/2019/12/elementwise-vs-matrix-multiplication.html


### Indexing and slicing
You can access and modify individual elements of a tensor using indexing and slicing, similar to how you would with a Python list or numpy array:
```Python
import torch

# Create a tensor
tensor = torch.Tensor([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Access a single element
print(tensor[0, 0])

# Modify a single element
tensor[0, 0] = 100
print(tensor[0,0])

# Access a row
print(tensor[1, :])

# Access a column
print(tensor[:, 2])

# Slice a tensor
print(tensor[1:3, 1:3])
```

### Data types and device
Tensors in PyTorch have a data type and can be move to different devices (such as CPUs and GPUs). You can specify the data type and device when creating a tensor, or you can use the to() method to convert a tensor to a different data type or device.
```Python
import torch

# Create a tensor with data type int32
tensor = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.int32)
print(tensor.dtype)

# Convert the tensor to data type float32
tensor = tensor.to(dtype=torch.float32)
print(tensor.dtype)

# Check if a tensor is on the CPU
print(tensor.is_cuda)

# Move the tensor to the GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = tensor.to(device)
print(tensor.is_cuda)
```

### Autograd
PyTorch provides automatic differentiation (autograd) for building and training neural networks. You can define a tensor as requiring gradients using the requires_grad flag, and then use the backward() method to compute the gradients with respect to the tensor.
```Python
import torch

# Create a tensor with requires_grad set to True
x = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)

# Perform a simple operation on the tensor
y = x + 2

# Compute the gradients with respect to x
y.backward(torch.ones_like(y))

# Print the gradients
print(x.grad)

```

### Neural Networks
You can use PyTorch to define and train neural networks. Here's an example of a simple fully-connected (linear) neural network for binary classification:
```Python
import torch
import torch.nn as nn

# Define the model
class Net(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes):
		super(Net, self).__init()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, num_classes)

	def forward(self, x):
		x = self.fc1(x)
		x = torch.relu(x)
		x = self.fc2(x)
		return x

# Create an instance of the model
model = Net(input_size=2, hidden_size=4, num_classes=2)

# Print the model architecture
print(model)
```

### Loss and Optimizer
To train a neural network, you need to define a loss function and an optimizer. PyTorch provides a variety of loss functions and optimizers that you can use. For example, here's how to use the cross-entropy loss and the Adam optmizer:
```Python
import torch
import torch.nn as nn

# Define the model (as before)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

model = Net(input_size=2, hidden_size=4, num_classes=2)

# Define the loss and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

```

### Training
To train a neural network, you need to loop through the training data, pass the input through the model to get the predictions, compute the loss, and then backpropogate the gradients and update the model's parameters using the optimizer. Here's an exmaple of a training loop for one epoch:
```Python
import torch

# Define the model, loss, and optimizer (as before)

# Get some dummy training data
inputs = torch.Tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
labels = torch.Tensor([0, 1, 1, 0]).long()

# Set the model to training mode
model.train()

# Loop through the training data
for i, (inputs, labels) in enumerate(zip(inputs, labels)):
    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # Backward pass and update
    loss.backward()
    optimizer.step()


```

You can repeat this loop for multiple epochs and use some kind of validation set to evaluate the model's performance and tune the hyperparamters.
