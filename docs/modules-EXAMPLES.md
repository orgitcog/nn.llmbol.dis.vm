# Usage Examples

This document provides practical examples for using the nn.llmbol.dis.vm system.

## Example 1: Simple Feedforward Network

```typescript
import { buildModel, createMLModule } from '~/lib/modules';

// Create a simple 3-layer feedforward network
const model = buildModel('mnist-classifier', [784])
  .linear(784, 256)
  .relu()
  .dropout(0.2)
  .linear(256, 128)
  .relu()
  .dropout(0.2)
  .linear(128, 10)
  .build();

// Create input (flattened 28x28 image)
const ml = createMLModule();
const input = ml.createTensor([1, 784], 'float32');

// Forward pass
const output = model.forward(input);

// Get predictions
const predictions = ml.softmax(output);
console.log('Predictions:', predictions);
```

## Example 2: Convolutional Network

```typescript
import { buildModel } from '~/lib/modules';

// Build a simple CNN
const model = buildModel('simple-cnn', [1, 28, 28])
  .conv1d(1, 16, 3)
  .relu()
  .batchNorm(16)
  .conv1d(16, 32, 3)
  .relu()
  .batchNorm(32)
  .linear(32, 10)
  .build();

console.log(model.summary());
```

## Example 3: Custom Architecture with Parallel Paths

```typescript
import { nn, parallel, linear, relu } from '~/lib/modules';

// Create a model with parallel paths
const mainPath = nn()
  .add(linear(128, 64))
  .add(relu())
  .add(linear(64, 32));

const skipPath = nn()
  .add(linear(128, 32));

// This is a simplified example - real implementation would
// need proper residual connections
```

## Example 4: Model Quantization

```typescript
import { createMLModule, Quantization } from '~/lib/modules';

const ml = createMLModule();

// Create a weight tensor
const weights = ml.createTensor([1024, 512], 'float32');

// Quantize to 8-bit
const quantized = Quantization.quantize(weights, {
  type: 'q8_0',
  blockSize: 32,
});

console.log('Original size:', weights.data.length * 4, 'bytes');
console.log('Quantized size:', quantized.data.length, 'bytes');

// Dequantize for inference
const dequantized = Quantization.dequantize(quantized, {
  type: 'q8_0',
  blockSize: 32,
});
```

## Example 5: Distributed Training Setup

```typescript
import { 
  createComputeNode, 
  createScheduler, 
  createTask,
  createCommunication 
} from '~/lib/modules';

// Setup distributed system
const setupDistributedSystem = () => {
  // Create compute nodes
  const nodes = [
    createComputeNode('gpu-1', '192.168.1.10', 8080, ['training', 'inference']),
    createComputeNode('gpu-2', '192.168.1.11', 8080, ['training', 'inference']),
    createComputeNode('cpu-1', '192.168.1.12', 8080, ['inference']),
  ];

  // Create scheduler with least-loaded strategy
  const scheduler = createScheduler({
    strategy: 'least-loaded',
    maxRetries: 3,
    timeout: 60000,
  });

  // Register nodes
  nodes.forEach(node => scheduler.registerNode(node));

  // Setup communication
  const comm = createCommunication('master');
  nodes.forEach(node => {
    const info = node.getInfo();
    comm.registerPeer(info.id);
  });
  comm.start();

  return { scheduler, nodes, comm };
};

// Submit training tasks
const submitTrainingBatch = (scheduler, data) => {
  const tasks = data.map((batch, i) => 
    createTask(`train-${i}`, 'training', { batch }, 1)
  );
  
  scheduler.submitTasks(tasks);
};

// Usage
const { scheduler, nodes, comm } = setupDistributedSystem();
const trainingData = [/* batch data */];
submitTrainingBatch(scheduler, trainingData);

// Monitor progress
setInterval(() => {
  console.log(scheduler.getStats());
  console.log(comm.getStats());
}, 5000);
```

## Example 6: VM Bytecode Execution

```typescript
import { createVM, BytecodeLoader, runtime } from '~/lib/modules';

// Create VM
const vm = createVM({
  maxMemory: 1024 * 1024 * 256, // 256MB
  maxThreads: 2,
  securityLevel: 'strict',
});

// Create custom bytecode
const code = [
  0x01, 0x0A, // PUSH 10
  0x01, 0x05, // PUSH 5
  0x00,       // NOP
  0xFF,       // HALT
];

const bytecode = BytecodeLoader.createModule(code);

// Load module
await vm.loadModule('calculator', bytecode);

// Create and execute process
const process = runtime.createProcess(1024 * 64);
const result = runtime.execute(process.id, bytecode);

console.log('Execution result:', result);
console.log('VM stats:', vm.getStats());
```

## Example 7: Inference Engine

```typescript
import { 
  createInferenceEngine, 
  createMLModule,
  type MLModel 
} from '~/lib/modules';

// Create mock model
const ml = createMLModule();
const model: MLModel = ml.loadModel('test-model', {
  vocabSize: 10000,
  hiddenSize: 768,
  numLayers: 12,
  numHeads: 12,
  maxSequenceLength: 512,
});

// Create inference engine
const engine = createInferenceEngine({
  batchSize: 1,
  maxTokens: 100,
  temperature: 0.8,
  topK: 50,
  topP: 0.95,
});

// Run inference
const prompt = [1, 2, 3, 4, 5]; // Token IDs
const output = await engine.infer(model, prompt);

console.log('Generated tokens:', output);

// Generate autoregressively
const generated = await engine.generate(model, prompt, 50);
console.log('Generated sequence:', generated);
```

## Example 8: Dynamic Model Building

```typescript
import { ModelBuilder } from '~/lib/modules';

// Build model from configuration
const buildFromConfig = (config: any) => {
  const builder = new ModelBuilder(config.name, config.inputShape);
  
  for (const layer of config.layers) {
    switch (layer.type) {
      case 'linear':
        builder.linear(layer.in, layer.out, layer.bias);
        break;
      case 'relu':
        builder.relu();
        break;
      case 'dropout':
        builder.dropout(layer.p);
        break;
    }
  }
  
  return builder.build();
};

// Configuration
const config = {
  name: 'dynamic-model',
  inputShape: [100],
  layers: [
    { type: 'linear', in: 100, out: 50, bias: true },
    { type: 'relu' },
    { type: 'dropout', p: 0.3 },
    { type: 'linear', in: 50, out: 10, bias: true },
  ],
};

const model = buildFromConfig(config);

// Serialize to JSON
const json = new ModelBuilder('test', [100])
  .linear(100, 50)
  .relu()
  .linear(50, 10)
  .toJSON();

console.log('Model architecture:', json);

// Load from JSON
const restored = ModelBuilder.fromJSON(json);
```

## Example 9: Tensor Operations

```typescript
import { createMLModule, TensorOps } from '~/lib/modules';

const ml = createMLModule();

// Create tensors
const a = ml.createTensor([3, 4], 'float32');
const b = ml.createTensor([3, 4], 'float32');

// Fill with data
(a.data as Float32Array).set([
  1, 2, 3, 4,
  5, 6, 7, 8,
  9, 10, 11, 12
]);

// Operations
const reshaped = TensorOps.reshape(a, [4, 3]);
const transposed = TensorOps.transpose(reshaped);
const scaled = TensorOps.scale(a, 2.0);
const multiplied = TensorOps.multiply(a, b);

// Statistics
const stats = TensorOps.stats(a);
console.log('Tensor stats:', stats);
```

## Example 10: Task Scheduling Strategies

```typescript
import { createScheduler, createComputeNode, createTask } from '~/lib/modules';

// Test different scheduling strategies
const testSchedulingStrategy = async (strategy: any) => {
  const scheduler = createScheduler({ strategy });
  
  // Create nodes
  const nodes = Array.from({ length: 3 }, (_, i) =>
    createComputeNode(`node-${i}`, 'localhost', 8080 + i)
  );
  
  nodes.forEach(node => scheduler.registerNode(node));
  
  // Submit tasks
  const tasks = Array.from({ length: 10 }, (_, i) =>
    createTask(`task-${i}`, 'inference', { id: i }, Math.random())
  );
  
  scheduler.submitTasks(tasks);
  
  // Wait and check stats
  await new Promise(resolve => setTimeout(resolve, 1000));
  console.log(`${strategy} stats:`, scheduler.getStats());
};

// Test all strategies
['round-robin', 'least-loaded', 'random', 'capability-based'].forEach(
  strategy => testSchedulingStrategy(strategy)
);
```

## Tips and Best Practices

1. **Memory Management**: Always configure appropriate memory limits for your VM based on model size
2. **Quantization**: Use Q8_0 for a good balance of speed and accuracy; use Q4_0 for maximum compression
3. **Distributed Computing**: Start with 'least-loaded' strategy for general purpose workloads
4. **Model Building**: Use `ModelBuilder` for complex architectures, `nn()` for quick prototypes
5. **Testing**: Always test bytecode modules before deployment
6. **Monitoring**: Regularly check stats from VM, scheduler, and communication modules
