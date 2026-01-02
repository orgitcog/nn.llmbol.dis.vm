# nn.llmbol.dis.vm - Neural Network VM Deployment System

A comprehensive system for deploying and executing neural network models with:
- **Inferno-inspired VM**: Bytecode-based deployment engine with `.dis` modules
- **GGML-like ML operations**: High-performance tensor operations with quantization support
- **Torch7-style architectures**: Modular neural network builder with dynamic composition
- **Distributed compute**: Task scheduling and inter-node communication

## Architecture

### VM Module (`diy.[+].dis`)
The VM deployment engine provides:
- Dynamic bytecode loading and execution
- Process management and isolation
- Module dependency resolution
- Memory management

### ML Module (`ml.[+].m`)
GGML-inspired machine learning operations:
- Tensor creation and manipulation
- Matrix operations (GEMM, element-wise ops)
- Activation functions (ReLU, Softmax, etc.)
- Model quantization (Q4_0, Q8_0)
- Inference engine

### NN Module (`nn.[+].b`)
Torch7-style neural network builder:
- Sequential and Parallel containers
- Common layers (Linear, Conv1d, etc.)
- Activation functions
- Batch normalization and dropout
- Dynamic model construction

### Distributed Module
Distributed computing capabilities:
- Compute node management
- Task scheduling with multiple strategies
- Inter-node communication
- Load balancing and fault tolerance

## Quick Start

### Creating a VM

```typescript
import { createVM, BytecodeLoader } from '~/lib/modules';

// Create a VM instance
const vm = createVM({
  maxMemory: 1024 * 1024 * 512,
  maxThreads: 4,
  enableDistributed: true,
});

// Create bytecode
const code = [0x00, 0x01, 0x05, 0xFF]; // NOP, PUSH 5, HALT
const bytecode = BytecodeLoader.createModule(code);

// Load and execute
await vm.loadModule('my-module', bytecode);
```

### Building Neural Networks

```typescript
import { buildModel, nn, linear, relu } from '~/lib/modules';

// Method 1: Using ModelBuilder
const model = buildModel('my-model', [784])
  .linear(784, 128)
  .relu()
  .dropout(0.2)
  .linear(128, 64)
  .relu()
  .linear(64, 10)
  .build();

// Method 2: Using Sequential directly
const model2 = nn()
  .add(linear(784, 128))
  .add(relu())
  .add(linear(128, 10));

// Forward pass
const input = {
  shape: [1, 784],
  data: new Float32Array(784),
  dtype: 'float32' as const,
};
const output = model.forward(input);

// Get model summary
console.log(model.summary());
```

### ML Operations

```typescript
import { createMLModule, Quantization } from '~/lib/modules';

const ml = createMLModule();

// Create tensors
const a = ml.createTensor([2, 3], 'float32');
const b = ml.createTensor([3, 2], 'float32');

// Matrix multiplication
const result = ml.matmul(a, b);

// Quantization
const quantized = Quantization.quantize(a, {
  type: 'q8_0',
  blockSize: 32,
});
```

### Distributed Computing

```typescript
import { 
  createComputeNode, 
  createScheduler, 
  createTask 
} from '~/lib/modules';

// Create compute nodes
const node1 = createComputeNode('node1', 'localhost', 8080, ['inference']);
const node2 = createComputeNode('node2', 'localhost', 8081, ['training']);

// Create scheduler
const scheduler = createScheduler({
  strategy: 'least-loaded',
  maxRetries: 3,
});

// Register nodes
scheduler.registerNode(node1);
scheduler.registerNode(node2);

// Submit tasks
const task = createTask('task1', 'inference', { data: 'test' }, 1);
scheduler.submitTask(task);

// Check stats
console.log(scheduler.getStats());
```

## Module Structure

```
app/lib/modules/
├── vm/                     # VM deployment engine
│   ├── diy.dis.ts         # Main VM implementation
│   ├── vm-runtime.ts      # Runtime execution environment
│   └── bytecode-loader.ts # Bytecode loading and validation
├── ml/                     # ML operations
│   ├── ml.m.ts            # Core ML module
│   ├── tensor-ops.ts      # Tensor operations
│   ├── quantization.ts    # Model quantization
│   └── inference-engine.ts # Inference execution
├── nn/                     # Neural network architecture
│   ├── nn.b.ts            # Build-a-bear network builder
│   ├── nn-modules.ts      # Common NN modules
│   ├── layer-factory.ts   # Layer factory
│   └── model-builder.ts   # Model builder
├── distributed/            # Distributed computing
│   ├── compute-node.ts    # Compute node
│   ├── task-scheduler.ts  # Task scheduler
│   └── communication.ts   # Inter-node communication
└── index.ts               # Main exports
```

## Features

### VM Features
- ✅ Dynamic module loading
- ✅ Process isolation
- ✅ Bytecode validation
- ✅ Memory management
- ✅ Module dependencies

### ML Features
- ✅ Tensor operations (reshape, transpose, concat, etc.)
- ✅ Matrix multiplication (GEMM)
- ✅ Activation functions (ReLU, Softmax, Tanh, Sigmoid)
- ✅ Model quantization (Q4_0, Q8_0)
- ✅ Inference engine
- ✅ Batch processing

### NN Features
- ✅ Sequential and Parallel containers
- ✅ Linear (fully connected) layers
- ✅ Convolutional layers (1D)
- ✅ Activation layers
- ✅ Dropout and Batch Normalization
- ✅ Dynamic model construction
- ✅ Model serialization

### Distributed Features
- ✅ Compute node management
- ✅ Multiple scheduling strategies (round-robin, least-loaded, random, capability-based)
- ✅ Task priority and retry
- ✅ Inter-node communication
- ✅ Heartbeat and health monitoring
- ✅ Load balancing

## API Reference

### VM API

```typescript
// Create VM
const vm = createVM(config?: Partial<VMConfig>): InfernoVM

// Load module
vm.loadModule(name: string, bytecode: Uint8Array): Promise<DisModule>

// Execute module
vm.execute(moduleName: string, entryPoint: string, args: any[]): Promise<any>

// Get stats
vm.getStats(): VMStats
```

### ML API

```typescript
// Create ML module
const ml = createMLModule(): MLModule

// Create tensor
ml.createTensor(shape: number[], dtype?: 'float32' | 'uint8' | 'int8'): Tensor

// Operations
ml.matmul(a: Tensor, b: Tensor): Tensor
ml.add(a: Tensor, b: Tensor): Tensor
ml.relu(input: Tensor): Tensor
ml.softmax(input: Tensor): Tensor
```

### NN API

```typescript
// Create model
const model = buildModel(name: string, inputShape: number[]): ModelBuilder

// Add layers
model.linear(inputSize: number, outputSize: number): ModelBuilder
model.relu(): ModelBuilder
model.dropout(p: number): ModelBuilder
model.build(): Sequential

// Forward pass
model.forward(input: Tensor): Tensor
```

### Distributed API

```typescript
// Create node
const node = createComputeNode(
  id: string, 
  address: string, 
  port: number, 
  capabilities?: string[]
): ComputeNode

// Create scheduler
const scheduler = createScheduler(config?: Partial<SchedulerConfig>): TaskScheduler

// Submit task
scheduler.submitTask(task: Task): string
```

## Testing

Tests are located in `app/lib/modules/__tests__/`:
- `vm.spec.ts` - VM module tests
- `ml.spec.ts` - ML operations tests
- `nn.spec.ts` - Neural network tests
- `distributed.spec.ts` - Distributed computing tests

Run tests with:
```bash
pnpm test app/lib/modules/__tests__
```

## Performance Considerations

- **Quantization**: Use Q4_0 or Q8_0 for reduced memory usage
- **Batch Processing**: Process multiple inputs together for better throughput
- **Distributed**: Enable distributed computing for large-scale workloads
- **Memory**: Configure VM memory limits based on model size

## License

MIT
