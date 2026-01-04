/**
 * nn.llmbol.dis.vm - Neural Network VM Deployment System
 * 
 * A comprehensive system for deploying and executing neural network models
 * with Inferno-inspired VM, GGML-like ML operations, Torch7-style architectures,
 * and distributed compute capabilities.
 */

// VM Module Exports
export { 
  InfernoVM, 
  createVM, 
  getGlobalVM,
  type DisModule,
  type VMConfig 
} from './vm/diy.dis';

export { 
  VMRuntime, 
  runtime,
  type VMProcess 
} from './vm/vm-runtime';

export { 
  BytecodeLoader,
  type BytecodeHeader 
} from './vm/bytecode-loader';

// ML Module Exports
export { 
  MLModule, 
  createMLModule, 
  getGlobalMLModule,
  type Tensor,
  type MLModel,
  type ModelConfig 
} from './ml/ml.m';

export { 
  TensorOps 
} from './ml/tensor-ops';

export { 
  Quantization,
  type QuantizationType,
  type QuantizationConfig 
} from './ml/quantization';

export { 
  InferenceEngine, 
  createInferenceEngine,
  type InferenceConfig 
} from './ml/inference-engine';

// NN Module Exports
export { 
  Sequential,
  Parallel,
  ConcatTable,
  nn,
  parallel,
  concatTable,
  type NNModule,
  type LinearConfig,
  type ConvConfig 
} from './nn/nn.b';

export { 
  Linear,
  ReLU,
  Tanh,
  Sigmoid,
  Dropout,
  BatchNorm,
  Conv1d 
} from './nn/nn-modules';

export { 
  LayerFactory,
  linear,
  relu,
  tanh,
  sigmoid,
  dropout,
  batchNorm,
  conv1d,
  type LayerType,
  type LayerConfig 
} from './nn/layer-factory';

export { 
  ModelBuilder,
  buildModel,
  createFeedforwardModel,
  createConvModel,
  type ModelArchitecture 
} from './nn/model-builder';

// Distributed Module Exports
export { 
  ComputeNode,
  createComputeNode,
  type NodeInfo,
  type Task 
} from './distributed/compute-node';

export { 
  TaskScheduler,
  createScheduler,
  createTask,
  type SchedulingStrategy,
  type SchedulerConfig 
} from './distributed/task-scheduler';

export { 
  Communication,
  createCommunication,
  type Message,
  type MessageType,
  type CommunicationConfig 
} from './distributed/communication';

/**
 * Version information
 */
export const VERSION = '1.0.0';

/**
 * System information
 */
export const SYSTEM_INFO = {
  name: 'nn.llmbol.dis.vm',
  version: VERSION,
  components: {
    vm: 'Inferno-inspired VM deployment engine',
    ml: 'GGML-like ML module implementations',
    nn: 'Torch7-style neural network architecture',
    distributed: 'Distributed compute capabilities',
  },
};
