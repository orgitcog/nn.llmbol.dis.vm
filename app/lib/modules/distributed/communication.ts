/**
 * Inter-Node Communication
 * 
 * Handles message passing and communication between distributed compute nodes
 */

export type MessageType = 'request' | 'response' | 'broadcast' | 'heartbeat';

export interface Message {
  id: string;
  type: MessageType;
  from: string;
  to: string | string[];
  payload: any;
  timestamp: number;
}

export interface CommunicationConfig {
  heartbeatInterval: number;
  messageTimeout: number;
  maxRetries: number;
}

export class Communication {
  private nodeId: string;
  private peers: Map<string, { lastSeen: number; status: 'online' | 'offline' }>;
  private messageHandlers: Map<MessageType, (msg: Message) => void>;
  private pendingMessages: Map<string, Message>;
  private config: CommunicationConfig;
  private heartbeatTimer?: NodeJS.Timeout;

  constructor(nodeId: string, config: Partial<CommunicationConfig> = {}) {
    this.nodeId = nodeId;
    this.peers = new Map();
    this.messageHandlers = new Map();
    this.pendingMessages = new Map();
    this.config = {
      heartbeatInterval: config.heartbeatInterval || 5000,
      messageTimeout: config.messageTimeout || 10000,
      maxRetries: config.maxRetries || 3,
    };
  }

  /**
   * Start communication
   */
  start(): void {
    this.startHeartbeat();
  }

  /**
   * Stop communication
   */
  stop(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = undefined;
    }
  }

  /**
   * Send a message
   */
  async send(to: string | string[], payload: any, type: MessageType = 'request'): Promise<Message> {
    const message: Message = {
      id: this.generateMessageId(),
      type,
      from: this.nodeId,
      to,
      payload,
      timestamp: Date.now(),
    };

    this.pendingMessages.set(message.id, message);

    // Simulate message transmission
    await this.transmit(message);

    return message;
  }

  /**
   * Broadcast a message to all peers
   */
  async broadcast(payload: any): Promise<void> {
    const peerIds = Array.from(this.peers.keys());
    
    const message: Message = {
      id: this.generateMessageId(),
      type: 'broadcast',
      from: this.nodeId,
      to: peerIds,
      payload,
      timestamp: Date.now(),
    };

    await this.transmit(message);
  }

  /**
   * Register a message handler
   */
  onMessage(type: MessageType, handler: (msg: Message) => void): void {
    this.messageHandlers.set(type, handler);
  }

  /**
   * Handle incoming message
   */
  async handleMessage(message: Message): Promise<void> {
    const handler = this.messageHandlers.get(message.type);
    if (handler) {
      handler(message);
    }

    // Update peer status
    if (message.from !== this.nodeId) {
      this.updatePeer(message.from);
    }
  }

  /**
   * Register a peer
   */
  registerPeer(peerId: string): void {
    this.peers.set(peerId, {
      lastSeen: Date.now(),
      status: 'online',
    });
  }

  /**
   * Unregister a peer
   */
  unregisterPeer(peerId: string): boolean {
    return this.peers.delete(peerId);
  }

  /**
   * Update peer last seen time
   */
  private updatePeer(peerId: string): void {
    const peer = this.peers.get(peerId);
    if (peer) {
      peer.lastSeen = Date.now();
      peer.status = 'online';
    } else {
      this.registerPeer(peerId);
    }
  }

  /**
   * Start heartbeat
   */
  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      this.sendHeartbeat();
      this.checkPeerStatus();
    }, this.config.heartbeatInterval);
  }

  /**
   * Send heartbeat to all peers
   */
  private async sendHeartbeat(): Promise<void> {
    const peerIds = Array.from(this.peers.keys());
    
    if (peerIds.length > 0) {
      await this.send(peerIds, { status: 'alive' }, 'heartbeat');
    }
  }

  /**
   * Check peer status and mark offline if needed
   */
  private checkPeerStatus(): void {
    const now = Date.now();
    const timeout = this.config.heartbeatInterval * 3;

    for (const [peerId, peer] of this.peers.entries()) {
      if (now - peer.lastSeen > timeout) {
        peer.status = 'offline';
      }
    }
  }

  /**
   * Transmit message (simulation)
   */
  private async transmit(message: Message): Promise<void> {
    // In a real implementation, this would use actual network transport
    // (WebSockets, HTTP, gRPC, etc.)
    
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, Math.random() * 10));
    
    // Remove from pending
    this.pendingMessages.delete(message.id);
  }

  /**
   * Generate unique message ID
   */
  private generateMessageId(): string {
    return `${this.nodeId}_${Date.now()}_${Math.random().toString(36).substring(7)}`;
  }

  /**
   * Get peer status
   */
  getPeerStatus(peerId: string): 'online' | 'offline' | 'unknown' {
    const peer = this.peers.get(peerId);
    return peer ? peer.status : 'unknown';
  }

  /**
   * Get all peers
   */
  getPeers(): string[] {
    return Array.from(this.peers.keys());
  }

  /**
   * Get online peers
   */
  getOnlinePeers(): string[] {
    return Array.from(this.peers.entries())
      .filter(([_, peer]) => peer.status === 'online')
      .map(([id, _]) => id);
  }

  /**
   * Get communication statistics
   */
  getStats() {
    const totalPeers = this.peers.size;
    const onlinePeers = this.getOnlinePeers().length;
    const offlinePeers = totalPeers - onlinePeers;

    return {
      nodeId: this.nodeId,
      totalPeers,
      onlinePeers,
      offlinePeers,
      pendingMessages: this.pendingMessages.size,
      config: this.config,
    };
  }
}

/**
 * Create a new communication instance
 */
export function createCommunication(
  nodeId: string,
  config?: Partial<CommunicationConfig>
): Communication {
  return new Communication(nodeId, config);
}
