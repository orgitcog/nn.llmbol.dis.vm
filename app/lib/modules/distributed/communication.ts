/**
 * Inter-Node Communication
 *
 * Handles message passing and communication between distributed compute nodes
 * with pluggable transport adapters.
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

/**
 * Pluggable transport backend for delivering messages between nodes.
 */
export interface TransportAdapter {
  /** Deliver `message` to one or more recipients. */
  send(to: string | string[], message: Message): Promise<void>;

  /** Register a handler invoked for every received message. */
  onReceive(handler: (message: Message) => void): void;

  /** Returns true when the transport is ready to send. */
  isConnected(): boolean;
}

/** In-memory fallback handler registry keyed by channel name. */
const _bcFallbackHandlers: Map<string, Array<(msg: Message) => void>> = new Map();

/**
 * Transport backed by the BroadcastChannel API.
 * Falls back to an in-memory handler map in environments without BroadcastChannel.
 */
export class BroadcastChannelTransport implements TransportAdapter {
  private _channel: BroadcastChannel | null = null;
  private _handler: ((msg: Message) => void) | null = null;
  private readonly _channelName: string;

  constructor(channelName: string = 'nn-vm-distributed') {
    this._channelName = channelName;

    if (typeof BroadcastChannel !== 'undefined') {
      this._channel = new BroadcastChannel(channelName);
    }
  }

  /** Broadcast the message over the channel (or in-memory fallback). */
  async send(_to: string | string[], message: Message): Promise<void> {
    if (this._channel) {
      this._channel.postMessage(JSON.stringify(message));
    } else {
      const handlers = _bcFallbackHandlers.get(this._channelName) ?? [];

      for (const h of handlers) {
        h(message);
      }
    }
  }

  /** Register a handler for incoming messages. */
  onReceive(handler: (message: Message) => void): void {
    this._handler = handler;

    if (this._channel) {
      this._channel.onmessage = (event: MessageEvent) => {
        try {
          const msg = JSON.parse(event.data as string) as Message;
          handler(msg);
        } catch {
          // ignore malformed messages
        }
      };
    } else {
      const existing = _bcFallbackHandlers.get(this._channelName) ?? [];
      existing.push(handler);
      _bcFallbackHandlers.set(this._channelName, existing);
    }
  }

  /** Returns true when the channel has been initialised. */
  isConnected(): boolean {
    return this._channel !== null || _bcFallbackHandlers.has(this._channelName);
  }
}

/**
 * Transport backed by a WebSocket connection.
 * Queues outbound messages while the socket is not yet open.
 */
export class WebSocketTransport implements TransportAdapter {
  private _ws: WebSocket | null = null;
  private _handler: ((msg: Message) => void) | null = null;
  private readonly _url: string;
  private _queue: Message[] = [];

  constructor(url: string) {
    this._url = url;
  }

  /** Open the WebSocket connection. No-op in environments without WebSocket. */
  connect(): void {
    if (typeof WebSocket === 'undefined') {
      this._ws = null;
      return;
    }

    this._ws = new WebSocket(this._url);

    this._ws.onmessage = (event: MessageEvent) => {
      if (this._handler) {
        try {
          const msg = JSON.parse(event.data as string) as Message;
          this._handler(msg);
        } catch {
          // ignore malformed messages
        }
      }
    };

    this._ws.onopen = () => {
      for (const msg of this._queue) {
        this._ws?.send(JSON.stringify(msg));
      }

      this._queue = [];
    };
  }

  /** Send via WebSocket if open, otherwise queue the message. */
  async send(_to: string | string[], message: Message): Promise<void> {
    if (this._ws && this._ws.readyState === WebSocket.OPEN) {
      this._ws.send(JSON.stringify(message));
    } else {
      this._queue.push(message);
    }
  }

  /** Register a handler for incoming messages. */
  onReceive(handler: (message: Message) => void): void {
    this._handler = handler;
  }

  /** Returns true when the WebSocket is in the OPEN state. */
  isConnected(): boolean {
    return this._ws !== null && this._ws.readyState === WebSocket.OPEN;
  }
}

export class Communication {
  private _nodeId: string;
  private _peers: Map<string, { lastSeen: number; status: 'online' | 'offline' }>;
  private _messageHandlers: Map<MessageType, (msg: Message) => void>;
  private _pendingMessages: Map<string, Message>;
  private _config: CommunicationConfig;
  private _heartbeatTimer?: ReturnType<typeof setInterval>;
  private _transport: TransportAdapter;

  constructor(nodeId: string, config: Partial<CommunicationConfig> = {}, transport?: TransportAdapter) {
    this._nodeId = nodeId;
    this._peers = new Map();
    this._messageHandlers = new Map();
    this._pendingMessages = new Map();
    this._config = {
      heartbeatInterval: config.heartbeatInterval ?? 5000,
      messageTimeout: config.messageTimeout ?? 10000,
      maxRetries: config.maxRetries ?? 3,
    };
    this._transport = transport ?? new BroadcastChannelTransport();
  }

  /**
   * Start communication and register the transport receive handler.
   */
  start(): void {
    this._transport.onReceive((msg) => this.handleMessage(msg));
    this._startHeartbeat();
  }

  /**
   * Stop communication and cancel the heartbeat timer.
   */
  stop(): void {
    if (this._heartbeatTimer !== undefined) {
      clearInterval(this._heartbeatTimer);
      this._heartbeatTimer = undefined;
    }
  }

  /**
   * Send a message to one or more peers.
   */
  async send(to: string | string[], payload: any, type: MessageType = 'request'): Promise<Message> {
    const message: Message = {
      id: this._generateMessageId(),
      type,
      from: this._nodeId,
      to,
      payload,
      timestamp: Date.now(),
    };

    this._pendingMessages.set(message.id, message);
    await this._transmit(message);

    return message;
  }

  /**
   * Broadcast a message to all known peers.
   */
  async broadcast(payload: any): Promise<void> {
    const peerIds = Array.from(this._peers.keys());

    const message: Message = {
      id: this._generateMessageId(),
      type: 'broadcast',
      from: this._nodeId,
      to: peerIds,
      payload,
      timestamp: Date.now(),
    };

    await this._transmit(message);
  }

  /**
   * Register a handler for a specific message type.
   */
  onMessage(type: MessageType, handler: (msg: Message) => void): void {
    this._messageHandlers.set(type, handler);
  }

  /**
   * Handle an incoming message, invoking the registered handler and updating peer state.
   */
  async handleMessage(message: Message): Promise<void> {
    const handler = this._messageHandlers.get(message.type);

    if (handler) {
      handler(message);
    }

    if (message.from !== this._nodeId) {
      this._updatePeer(message.from);
    }
  }

  /**
   * Register a peer node by ID.
   */
  registerPeer(peerId: string): void {
    this._peers.set(peerId, { lastSeen: Date.now(), status: 'online' });
  }

  /**
   * Unregister a peer node. Returns true if it existed.
   */
  unregisterPeer(peerId: string): boolean {
    return this._peers.delete(peerId);
  }

  /**
   * Get the current status of a peer.
   */
  getPeerStatus(peerId: string): 'online' | 'offline' | 'unknown' {
    const peer = this._peers.get(peerId);
    return peer ? peer.status : 'unknown';
  }

  /**
   * Get all registered peer IDs.
   */
  getPeers(): string[] {
    return Array.from(this._peers.keys());
  }

  /**
   * Get IDs of all peers currently marked online.
   */
  getOnlinePeers(): string[] {
    return Array.from(this._peers.entries())
      .filter(([, peer]) => peer.status === 'online')
      .map(([id]) => id);
  }

  /**
   * Get communication statistics for this node.
   */
  getStats() {
    const totalPeers = this._peers.size;
    const onlinePeers = this.getOnlinePeers().length;
    const offlinePeers = totalPeers - onlinePeers;

    return {
      nodeId: this._nodeId,
      totalPeers,
      onlinePeers,
      offlinePeers,
      pendingMessages: this._pendingMessages.size,
      config: this._config,
    };
  }

  private _updatePeer(peerId: string): void {
    const peer = this._peers.get(peerId);

    if (peer) {
      peer.lastSeen = Date.now();
      peer.status = 'online';
    } else {
      this.registerPeer(peerId);
    }
  }

  private _startHeartbeat(): void {
    this._heartbeatTimer = setInterval(() => {
      this._sendHeartbeat();
      this._checkPeerStatus();
    }, this._config.heartbeatInterval);
  }

  private async _sendHeartbeat(): Promise<void> {
    const peerIds = Array.from(this._peers.keys());

    if (peerIds.length > 0) {
      await this.send(peerIds, { status: 'alive' }, 'heartbeat');
    }
  }

  private _checkPeerStatus(): void {
    const now = Date.now();
    const timeout = this._config.heartbeatInterval * 3;

    for (const [, peer] of this._peers.entries()) {
      if (now - peer.lastSeen > timeout) {
        peer.status = 'offline';
      }
    }
  }

  private async _transmit(message: Message): Promise<void> {
    let lastError: Error | undefined;

    for (let attempt = 0; attempt <= this._config.maxRetries; attempt++) {
      try {
        await Promise.race([
          this._transport.send(message.to, message),
          new Promise<never>((_, reject) =>
            setTimeout(() => reject(new Error('Message timeout')), this._config.messageTimeout),
          ),
        ]);
        this._pendingMessages.delete(message.id);

        return;
      } catch (err) {
        lastError = err instanceof Error ? err : new Error(String(err));

        if (attempt < this._config.maxRetries) {
          await new Promise((resolve) => setTimeout(resolve, 100));
        }
      }
    }

    this._pendingMessages.delete(message.id);
    throw lastError;
  }

  private _generateMessageId(): string {
    return `${this._nodeId}_${Date.now()}_${Math.random().toString(36).substring(7)}`;
  }
}

/**
 * Create a new Communication instance with the default BroadcastChannel transport.
 */
export function createCommunication(nodeId: string, config?: Partial<CommunicationConfig>): Communication {
  return new Communication(nodeId, config);
}

/**
 * Create a new Communication instance with a custom transport adapter.
 */
export function createCommunicationWithTransport(
  nodeId: string,
  transport: TransportAdapter,
  config?: Partial<CommunicationConfig>,
): Communication {
  return new Communication(nodeId, config, transport);
}
