/**
 * VM Runtime Execution Environment
 *
 * Provides the runtime environment for executing bytecode and managing
 * VM state, memory, and process scheduling.
 *
 * Instruction set:
 *   0x00 NOP        – no operation
 *   0x01 PUSH       – push float32 (next 4 bytes, LE)
 *   0x02 POP        – discard top of stack
 *   0x03 PUSH_INT   – push uint8 (next 1 byte)
 *   0x10 ADD        – pop b, pop a, push a+b
 *   0x11 SUB        – pop b, pop a, push a-b
 *   0x12 MUL        – pop b, pop a, push a*b
 *   0x13 DIV        – pop b, pop a, push a/b (throws on b==0)
 *   0x20 EQ         – pop b, pop a, push (a===b ? 1 : 0)
 *   0x21 LT         – pop b, pop a, push (a<b  ? 1 : 0)
 *   0x22 GT         – pop b, pop a, push (a>b  ? 1 : 0)
 *   0x30 JMP        – next 2 bytes = uint16 target; jump to it
 *   0x31 JZ         – next 2 bytes = target; pop top, jump if === 0
 *   0x32 JNZ        – next 2 bytes = target; pop top, jump if !== 0
 *   0x40 LOAD       – pop addr, push float32 from process.memory[addr]
 *   0x41 STORE      – pop addr, pop value, write float32 to process.memory[addr]
 *   0x50 DUP        – duplicate top of stack
 *   0x51 SWAP       – swap top two stack items
 *   0x60 CALL       – next 2 bytes = fn address; push return-addr; jump
 *   0x61 RET        – pop return address from call stack; jump to it
 *   0xFF HALT       – stop execution, return top of stack (or null)
 */

export interface VMProcess {
  id: string;
  state: 'running' | 'suspended' | 'terminated';
  memory: Uint8Array;
  registers: Map<string, unknown>;
  stackPointer: number;
}

export class VMRuntime {
  private _processes: Map<string, VMProcess>;
  private _nextProcessId: number;
  private _maxCycles: number;

  constructor(maxCycles: number = 1_000_000) {
    this._processes = new Map();
    this._nextProcessId = 0;
    this._maxCycles = maxCycles;
  }

  /**
   * Update the maximum number of instructions that may execute before an
   * error is thrown.  Useful for tests or resource-constrained environments.
   */
  setMaxCycles(n: number): void {
    this._maxCycles = n;
  }

  /**
   * Create a new VM process with its own isolated memory region.
   */
  createProcess(memorySize: number = 1024 * 64): VMProcess {
    const proc: VMProcess = {
      id: `proc_${this._nextProcessId++}`,
      state: 'suspended',
      memory: new Uint8Array(memorySize),
      registers: new Map(),
      stackPointer: 0,
    };

    this._processes.set(proc.id, proc);

    return proc;
  }

  /**
   * Execute bytecode in the named process.
   * The process starts at the entryPoint encoded in the 24-byte header, unless
   * the `__startPC` register has been set (used by module-export wrappers).
   */
  execute(processId: string, bytecode: Uint8Array): unknown {
    const proc = this._processes.get(processId);

    if (!proc) {
      throw new Error(`Process ${processId} not found`);
    }

    proc.state = 'running';

    try {
      const result = this._interpretBytecode(proc, bytecode);
      proc.state = 'suspended';

      return result;
    } catch (error) {
      proc.state = 'terminated';
      throw error;
    }
  }

  /**
   * Terminate and remove a process from the runtime.
   * Returns true if the process existed and was removed.
   */
  terminateProcess(processId: string): boolean {
    const proc = this._processes.get(processId);

    if (proc) {
      proc.state = 'terminated';

      return this._processes.delete(processId);
    }

    return false;
  }

  /**
   * Return the current state of a process, or undefined if not found.
   */
  getProcessStatus(processId: string): VMProcess | undefined {
    return this._processes.get(processId);
  }

  /**
   * Return a snapshot of all currently registered processes.
   */
  getAllProcesses(): VMProcess[] {
    return Array.from(this._processes.values());
  }

  // ── interpreter ────────────────────────────────────────────────────────────

  private _interpretBytecode(proc: VMProcess, bytecode: Uint8Array): unknown {
    const view = new DataView(bytecode.buffer, bytecode.byteOffset, bytecode.byteLength);
    const stack: number[] = [];
    const callStack: number[] = [];
    let cycles = 0;

    /*
     * Honour explicit start-PC override (set by export wrappers), otherwise
     * jump to the entryPoint stored in the 24-byte header.
     */
    let pc: number;

    if (proc.registers.has('__startPC')) {
      pc = proc.registers.get('__startPC') as number;
      proc.registers.delete('__startPC');
    } else {
      pc = bytecode.length >= 24 ? view.getUint32(8, true) : 0;
    }

    while (pc < bytecode.length) {
      if (++cycles > this._maxCycles) {
        throw new Error('Cycle budget exceeded');
      }

      const opcode = bytecode[pc++];

      switch (opcode) {
        case 0x00: // NOP
          break;

        case 0x01: {
          // PUSH float32
          const val = view.getFloat32(pc, true);
          pc += 4;
          stack.push(val);
          break;
        }

        case 0x02: // POP
          stack.pop();
          break;

        case 0x03: // PUSH_INT uint8
          stack.push(bytecode[pc++]!);
          break;

        case 0x10: {
          // ADD
          const b = stack.pop() ?? 0;
          const a = stack.pop() ?? 0;
          stack.push(a + b);
          break;
        }

        case 0x11: {
          // SUB
          const b = stack.pop() ?? 0;
          const a = stack.pop() ?? 0;
          stack.push(a - b);
          break;
        }

        case 0x12: {
          // MUL
          const b = stack.pop() ?? 0;
          const a = stack.pop() ?? 0;
          stack.push(a * b);
          break;
        }

        case 0x13: {
          // DIV
          const b = stack.pop() ?? 0;
          const a = stack.pop() ?? 0;

          if (b === 0) {
            throw new Error('Division by zero');
          }

          stack.push(a / b);
          break;
        }

        case 0x20: {
          // EQ
          const b = stack.pop() ?? 0;
          const a = stack.pop() ?? 0;
          stack.push(a === b ? 1 : 0);
          break;
        }

        case 0x21: {
          // LT
          const b = stack.pop() ?? 0;
          const a = stack.pop() ?? 0;
          stack.push(a < b ? 1 : 0);
          break;
        }

        case 0x22: {
          // GT
          const b = stack.pop() ?? 0;
          const a = stack.pop() ?? 0;
          stack.push(a > b ? 1 : 0);
          break;
        }

        case 0x30: {
          // JMP
          pc = view.getUint16(pc, true);
          break;
        }

        case 0x31: {
          // JZ
          const target = view.getUint16(pc, true);
          pc += 2;

          const jzTop = stack.pop() ?? 0;

          if (jzTop === 0) {
            pc = target;
          }

          break;
        }

        case 0x32: {
          // JNZ
          const target = view.getUint16(pc, true);
          pc += 2;

          const jnzTop = stack.pop() ?? 0;

          if (jnzTop !== 0) {
            pc = target;
          }

          break;
        }

        case 0x40: {
          // LOAD
          const addr = stack.pop() ?? 0;
          const memView = new DataView(proc.memory.buffer);
          stack.push(memView.getFloat32(addr, true));
          break;
        }

        case 0x41: {
          // STORE
          const addr = stack.pop() ?? 0;
          const storeVal = stack.pop() ?? 0;
          const memView = new DataView(proc.memory.buffer);
          memView.setFloat32(addr, storeVal, true);
          break;
        }

        case 0x50: // DUP
          if (stack.length > 0) {
            stack.push(stack[stack.length - 1]!);
          }

          break;

        case 0x51: {
          // SWAP
          if (stack.length >= 2) {
            const swapB = stack.pop()!;
            const swapA = stack.pop()!;
            stack.push(swapB);
            stack.push(swapA);
          }

          break;
        }

        case 0x60: {
          // CALL
          const fnAddr = view.getUint16(pc, true);
          pc += 2;
          callStack.push(pc); // return address = byte after the CALL instruction
          pc = fnAddr;
          break;
        }

        case 0x61: {
          // RET
          const retAddr = callStack.pop();

          if (retAddr === undefined) {
            // Returning from the top-level call frame
            return stack.length > 0 ? stack[stack.length - 1] : null;
          }

          pc = retAddr;
          break;
        }

        case 0xff: // HALT
          return stack.length > 0 ? stack[stack.length - 1] : null;

        default:
          break;
      }

      proc.stackPointer = stack.length;
    }

    return stack.length > 0 ? stack[stack.length - 1] : null;
  }
}

export const runtime = new VMRuntime();
