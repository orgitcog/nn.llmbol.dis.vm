/**
 * Distributed Compute Node
 * 
 * Represents a compute node in a distributed system that can execute
 * tasks and communicate with other nodes
 */

export interface NodeInfo {
  id: string;
  address: string;
  port: number;
  status: 'idle' | 'busy' | 'offline';
  capabilities: string[];
  load: number;
}

export interface Task {
  id: string;
  type: string;
  payload: any;
  priority: number;
  createdAt: number;
  assignedTo?: string;
}

export class ComputeNode {
  private info: NodeInfo;
  private tasks: Map<string, Task>;
  private maxConcurrentTasks: number;

  constructor(id: string, address: string, port: number, capabilities: string[] = []) {
    this.info = {
      id,
      address,
      port,
      status: 'idle',
      capabilities,
      load: 0,
    };
    this.tasks = new Map();
    this.maxConcurrentTasks = 4;
  }

  /**
   * Get node information
   */
  getInfo(): NodeInfo {
    return { ...this.info };
  }

  /**
   * Update node status
   */
  setStatus(status: NodeInfo['status']): void {
    this.info.status = status;
  }

  /**
   * Check if node can accept more tasks
   */
  canAcceptTask(): boolean {
    return this.info.status !== 'offline' && this.tasks.size < this.maxConcurrentTasks;
  }

  /**
   * Assign a task to this node
   */
  assignTask(task: Task): boolean {
    if (!this.canAcceptTask()) {
      return false;
    }

    task.assignedTo = this.info.id;
    this.tasks.set(task.id, task);
    this.updateLoad();
    
    if (this.tasks.size > 0) {
      this.info.status = 'busy';
    }

    return true;
  }

  /**
   * Execute a task
   */
  async executeTask(taskId: string): Promise<any> {
    const task = this.tasks.get(taskId);
    if (!task) {
      throw new Error(`Task ${taskId} not found`);
    }

    try {
      // Simulate task execution
      const result = await this.processTask(task);
      this.tasks.delete(taskId);
      this.updateLoad();
      
      if (this.tasks.size === 0) {
        this.info.status = 'idle';
      }

      return result;
    } catch (error) {
      this.tasks.delete(taskId);
      this.updateLoad();
      throw error;
    }
  }

  /**
   * Process a task based on its type
   */
  private async processTask(task: Task): Promise<any> {
    // This is where actual task processing would happen
    // Different task types would be handled differently
    
    switch (task.type) {
      case 'inference':
        return this.processInference(task.payload);
      case 'training':
        return this.processTraining(task.payload);
      case 'data_processing':
        return this.processData(task.payload);
      default:
        return { success: true, taskId: task.id };
    }
  }

  /**
   * Process inference task
   */
  private async processInference(payload: any): Promise<any> {
    // Simulate inference computation
    await this.delay(100);
    return { type: 'inference', result: payload };
  }

  /**
   * Process training task
   */
  private async processTraining(payload: any): Promise<any> {
    // Simulate training computation
    await this.delay(500);
    return { type: 'training', result: payload };
  }

  /**
   * Process data task
   */
  private async processData(payload: any): Promise<any> {
    // Simulate data processing
    await this.delay(50);
    return { type: 'data_processing', result: payload };
  }

  /**
   * Update node load
   */
  private updateLoad(): void {
    this.info.load = this.tasks.size / this.maxConcurrentTasks;
  }

  /**
   * Helper delay function
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Get current tasks
   */
  getTasks(): Task[] {
    return Array.from(this.tasks.values());
  }

  /**
   * Get task count
   */
  getTaskCount(): number {
    return this.tasks.size;
  }

  /**
   * Get node load
   */
  getLoad(): number {
    return this.info.load;
  }

  /**
   * Check if node has specific capability
   */
  hasCapability(capability: string): boolean {
    return this.info.capabilities.includes(capability);
  }

  /**
   * Add capability
   */
  addCapability(capability: string): void {
    if (!this.hasCapability(capability)) {
      this.info.capabilities.push(capability);
    }
  }

  /**
   * Remove capability
   */
  removeCapability(capability: string): void {
    const index = this.info.capabilities.indexOf(capability);
    if (index > -1) {
      this.info.capabilities.splice(index, 1);
    }
  }
}

/**
 * Create a new compute node
 */
export function createComputeNode(
  id: string,
  address: string = 'localhost',
  port: number = 8080,
  capabilities: string[] = []
): ComputeNode {
  return new ComputeNode(id, address, port, capabilities);
}
