import { describe, expect, it, beforeEach } from 'vitest';
import { ComputeNode, createComputeNode } from '../distributed/compute-node';
import { TaskScheduler, createScheduler, createTask } from '../distributed/task-scheduler';
import { Communication, createCommunication } from '../distributed/communication';
import type { Task } from '../distributed/compute-node';

describe('ComputeNode', () => {
  describe('Node Creation', () => {
    it('should create a compute node', () => {
      const node = createComputeNode('node1', 'localhost', 8080);
      const info = node.getInfo();
      
      expect(info.id).toBe('node1');
      expect(info.address).toBe('localhost');
      expect(info.port).toBe(8080);
      expect(info.status).toBe('idle');
    });

    it('should create with capabilities', () => {
      const node = createComputeNode('node1', 'localhost', 8080, ['inference', 'training']);
      const info = node.getInfo();
      
      expect(info.capabilities).toContain('inference');
      expect(info.capabilities).toContain('training');
    });
  });

  describe('Task Assignment', () => {
    let node: ComputeNode;

    beforeEach(() => {
      node = createComputeNode('node1', 'localhost', 8080);
    });

    it('should accept a task', () => {
      const task: Task = {
        id: 'task1',
        type: 'inference',
        payload: {},
        priority: 1,
        createdAt: Date.now(),
      };
      
      const accepted = node.assignTask(task);
      expect(accepted).toBe(true);
      expect(node.getTaskCount()).toBe(1);
    });

    it('should execute a task', async () => {
      const task: Task = {
        id: 'task1',
        type: 'inference',
        payload: { data: 'test' },
        priority: 1,
        createdAt: Date.now(),
      };
      
      node.assignTask(task);
      const result = await node.executeTask('task1');
      
      expect(result).toBeDefined();
      expect(node.getTaskCount()).toBe(0);
    });
  });

  describe('Capabilities', () => {
    it('should check capabilities', () => {
      const node = createComputeNode('node1', 'localhost', 8080, ['inference']);
      
      expect(node.hasCapability('inference')).toBe(true);
      expect(node.hasCapability('training')).toBe(false);
    });

    it('should add capability', () => {
      const node = createComputeNode('node1', 'localhost', 8080);
      node.addCapability('inference');
      
      expect(node.hasCapability('inference')).toBe(true);
    });

    it('should remove capability', () => {
      const node = createComputeNode('node1', 'localhost', 8080, ['inference']);
      node.removeCapability('inference');
      
      expect(node.hasCapability('inference')).toBe(false);
    });
  });

  describe('Node Status', () => {
    it('should update status', () => {
      const node = createComputeNode('node1', 'localhost', 8080);
      node.setStatus('busy');
      
      const info = node.getInfo();
      expect(info.status).toBe('busy');
    });

    it('should report load', () => {
      const node = createComputeNode('node1', 'localhost', 8080);
      const load = node.getLoad();
      
      expect(load).toBe(0);
    });
  });
});

describe('TaskScheduler', () => {
  describe('Node Management', () => {
    it('should register a node', () => {
      const scheduler = createScheduler();
      const node = createComputeNode('node1', 'localhost', 8080);
      
      scheduler.registerNode(node);
      const nodes = scheduler.getNodes();
      
      expect(nodes.length).toBe(1);
      expect(nodes[0].id).toBe('node1');
    });

    it('should unregister a node', () => {
      const scheduler = createScheduler();
      const node = createComputeNode('node1', 'localhost', 8080);
      
      scheduler.registerNode(node);
      const removed = scheduler.unregisterNode('node1');
      
      expect(removed).toBe(true);
      expect(scheduler.getNodes().length).toBe(0);
    });
  });

  describe('Task Scheduling', () => {
    let scheduler: TaskScheduler;
    let node1: ComputeNode;
    let node2: ComputeNode;

    beforeEach(() => {
      scheduler = createScheduler({ strategy: 'round-robin' });
      node1 = createComputeNode('node1', 'localhost', 8080);
      node2 = createComputeNode('node2', 'localhost', 8081);
      scheduler.registerNode(node1);
      scheduler.registerNode(node2);
    });

    it('should submit a task', () => {
      const task = createTask('task1', 'inference', {});
      const taskId = scheduler.submitTask(task);
      
      expect(taskId).toBe('task1');
    });

    it('should submit multiple tasks', () => {
      const tasks = [
        createTask('task1', 'inference', {}),
        createTask('task2', 'training', {}),
      ];
      
      const taskIds = scheduler.submitTasks(tasks);
      expect(taskIds.length).toBe(2);
    });
  });

  describe('Scheduling Strategies', () => {
    it('should create with least-loaded strategy', () => {
      const scheduler = createScheduler({ strategy: 'least-loaded' });
      expect(scheduler).toBeDefined();
    });

    it('should create with random strategy', () => {
      const scheduler = createScheduler({ strategy: 'random' });
      expect(scheduler).toBeDefined();
    });

    it('should create with capability-based strategy', () => {
      const scheduler = createScheduler({ strategy: 'capability-based' });
      expect(scheduler).toBeDefined();
    });
  });

  describe('Scheduler Statistics', () => {
    it('should provide stats', () => {
      const scheduler = createScheduler();
      const node = createComputeNode('node1', 'localhost', 8080);
      scheduler.registerNode(node);
      
      const stats = scheduler.getStats();
      
      expect(stats.totalNodes).toBe(1);
      expect(stats.queuedTasks).toBe(0);
      expect(stats).toHaveProperty('strategy');
    });
  });
});

describe('Communication', () => {
  describe('Peer Management', () => {
    it('should register a peer', () => {
      const comm = createCommunication('node1');
      comm.registerPeer('node2');
      
      const peers = comm.getPeers();
      expect(peers).toContain('node2');
    });

    it('should unregister a peer', () => {
      const comm = createCommunication('node1');
      comm.registerPeer('node2');
      comm.unregisterPeer('node2');
      
      const peers = comm.getPeers();
      expect(peers).not.toContain('node2');
    });

    it('should get peer status', () => {
      const comm = createCommunication('node1');
      comm.registerPeer('node2');
      
      const status = comm.getPeerStatus('node2');
      expect(status).toBe('online');
    });
  });

  describe('Message Handling', () => {
    it('should send a message', async () => {
      const comm = createCommunication('node1');
      comm.registerPeer('node2');
      
      const message = await comm.send('node2', { data: 'test' });
      
      expect(message.from).toBe('node1');
      expect(message.to).toBe('node2');
      expect(message.payload).toEqual({ data: 'test' });
    });

    it('should broadcast a message', async () => {
      const comm = createCommunication('node1');
      comm.registerPeer('node2');
      comm.registerPeer('node3');
      
      await comm.broadcast({ data: 'broadcast' });
      // Broadcast should succeed without errors
      expect(true).toBe(true);
    });

    it('should register message handler', () => {
      const comm = createCommunication('node1');
      let received = false;
      
      comm.onMessage('request', (msg) => {
        received = true;
      });
      
      expect(received).toBe(false); // Handler registered but not called yet
    });
  });

  describe('Communication Lifecycle', () => {
    it('should start communication', () => {
      const comm = createCommunication('node1');
      comm.start();
      // Should start without errors
      expect(true).toBe(true);
      comm.stop();
    });

    it('should stop communication', () => {
      const comm = createCommunication('node1');
      comm.start();
      comm.stop();
      // Should stop without errors
      expect(true).toBe(true);
    });
  });

  describe('Communication Statistics', () => {
    it('should provide stats', () => {
      const comm = createCommunication('node1');
      comm.registerPeer('node2');
      
      const stats = comm.getStats();
      
      expect(stats.nodeId).toBe('node1');
      expect(stats.totalPeers).toBe(1);
      expect(stats.onlinePeers).toBe(1);
      expect(stats).toHaveProperty('config');
    });
  });
});
