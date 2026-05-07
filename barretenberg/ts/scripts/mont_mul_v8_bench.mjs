#!/usr/bin/env node

// Microbenchmark harness for the bn254 Fr/Fq Montgomery multiplication WASM
// exports declared in `barretenberg/cpp/src/barretenberg/ecc/mont_mul_bench_exports.cpp`.
//
// What this measures:
//   per-iteration time = JS loop body + JS arrow wrapper + JS->wasm boundary + kernel.
//   "ns/mul" is therefore (boundary + kernel) / ops-per-call, not the pure kernel.
//   The boundary is identical across wasm builds on the same machine, so this is
//   sound for *relative* comparisons (build A vs build B). Do not quote the
//   absolute number as the kernel cost in isolation.
//
// Wasm tier determinism:
//   Run with `--allow-natives-syntax --no-liftoff` (see package.json
//   "bench:montmul:v8" script). `--no-liftoff` disables V8's wasm baseline tier so
//   the wasm body compiles directly with TurboFan from the first call. The
//   `%PrepareFunctionForOptimization` / `%OptimizeFunctionOnNextCall` calls in
//   `warmUpRunner` only optimize the JS arrow wrapper, not the wasm code; the
//   "wrapper TF status" line below reflects the JS wrapper's tier.
//
// Artifact selection:
//   The harness uses a single canonical build directory (`build-wasm-threads-simd`)
//   and rebuilds `barretenberg.wasm` via `cmake --build` before every run, so the
//   binary under test always matches the current source tree. Pass `--wasm <path>`
//   to bypass the rebuild and load a specific artifact instead — useful for
//   comparing two pre-built wasm files (e.g. against a different git commit).
//   Pass `--no-rebuild` to skip the rebuild but keep the canonical path.

import { spawnSync } from 'node:child_process';
import { randomFillSync } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const FIELD_CONFIGS = {
  fr: {
    label: 'BN254 Fr',
    init: 'wasm_bench_bn254_fr_init_mont_mul_inputs',
    single: 'wasm_bench_bn254_fr_mont_mul',
    paired: 'wasm_bench_bn254_fr_paired_mont_mul',
  },
  fq: {
    label: 'BN254 Fq',
    init: 'wasm_bench_bn254_fq_init_mont_mul_inputs',
    single: 'wasm_bench_bn254_fq_mont_mul',
    paired: 'wasm_bench_bn254_fq_paired_mont_mul',
  },
};

const DEFAULT_ITERATIONS = 1_000_000;
const DEFAULT_WARMUP = 100_000;
const DEFAULT_INITIAL_PAGES = 35;
const DEFAULT_MAX_PAGES = 2 ** 16;

const CANONICAL_BUILD_DIR = path.resolve(__dirname, '../../cpp/build-wasm-threads-simd');
const CANONICAL_WASM_PATH = path.join(CANONICAL_BUILD_DIR, 'bin', 'barretenberg.wasm');

function parseArgs(argv) {
  const options = {
    field: 'all',
    iterations: DEFAULT_ITERATIONS,
    warmup: DEFAULT_WARMUP,
    wasmPath: undefined,
    rebuild: true,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--field' && argv[i + 1]) {
      options.field = argv[++i];
      continue;
    }
    if (arg.startsWith('--field=')) {
      options.field = arg.slice('--field='.length);
      continue;
    }
    if (arg === '--iterations' && argv[i + 1]) {
      options.iterations = parsePositiveInt(argv[++i], '--iterations');
      continue;
    }
    if (arg.startsWith('--iterations=')) {
      options.iterations = parsePositiveInt(arg.slice('--iterations='.length), '--iterations');
      continue;
    }
    if (arg === '--warmup' && argv[i + 1]) {
      options.warmup = parseNonNegativeInt(argv[++i], '--warmup');
      continue;
    }
    if (arg.startsWith('--warmup=')) {
      options.warmup = parseNonNegativeInt(arg.slice('--warmup='.length), '--warmup');
      continue;
    }
    if (arg === '--wasm' && argv[i + 1]) {
      options.wasmPath = argv[++i];
      options.rebuild = false;
      continue;
    }
    if (arg.startsWith('--wasm=')) {
      options.wasmPath = arg.slice('--wasm='.length);
      options.rebuild = false;
      continue;
    }
    if (arg === '--no-rebuild') {
      options.rebuild = false;
      continue;
    }
    if (arg === '--help' || arg === '-h') {
      printHelp();
      process.exit(0);
    }
    throw new Error(`Unknown argument: ${arg}`);
  }

  return options;
}

function parsePositiveInt(value, flagName) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${flagName} must be a positive integer, received: ${value}`);
  }
  return parsed;
}

function parseNonNegativeInt(value, flagName) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`${flagName} must be a non-negative integer, received: ${value}`);
  }
  return parsed;
}

function printHelp() {
  console.log(`Usage: node --allow-natives-syntax ./scripts/mont_mul_v8_bench.mjs [options]

Options:
  --field <fr|fq|all>      Which field export set to benchmark. Default: all
  --iterations <count>     Timed iterations per benchmark. Default: ${DEFAULT_ITERATIONS}
  --warmup <count>         Warmup iterations before timing. Default: ${DEFAULT_WARMUP}
  --wasm <path>            Load this wasm artifact instead of rebuilding the
                           canonical one. Disables the rebuild step.
  --no-rebuild             Skip the rebuild step but still load the canonical
                           artifact at ${CANONICAL_WASM_PATH}.

By default the script rebuilds the canonical wasm via:
  cmake --build ${CANONICAL_BUILD_DIR} --target barretenberg.wasm
`);
}

function rebuildCanonicalWasm() {
  if (!existsSync(CANONICAL_BUILD_DIR)) {
    throw new Error(
      `Canonical build directory not configured: ${CANONICAL_BUILD_DIR}\n` +
        `Configure it once with: cmake --preset wasm-threads-simd -S ${path.resolve(CANONICAL_BUILD_DIR, '..')}`,
    );
  }
  console.log(`Rebuilding ${CANONICAL_WASM_PATH} ...`);
  const result = spawnSync(
    'cmake',
    ['--build', CANONICAL_BUILD_DIR, '--target', 'barretenberg.wasm'],
    { stdio: 'inherit' },
  );
  if (result.status !== 0) {
    throw new Error(`cmake --build failed with exit code ${result.status}`);
  }
  if (!existsSync(CANONICAL_WASM_PATH)) {
    throw new Error(`Build succeeded but ${CANONICAL_WASM_PATH} is missing`);
  }
}

function resolveWasmPath(explicitPath) {
  if (explicitPath) {
    const resolved = path.resolve(process.cwd(), explicitPath);
    if (!existsSync(resolved)) {
      throw new Error(`WASM file not found: ${resolved}`);
    }
    return resolved;
  }
  if (!existsSync(CANONICAL_WASM_PATH)) {
    throw new Error(
      `Canonical wasm not present at ${CANONICAL_WASM_PATH}.\n` +
        `Either run with default rebuild enabled, or pass --wasm <path> to load a specific artifact.`,
    );
  }
  return CANONICAL_WASM_PATH;
}

function loadWasmBytes(wasmPath) {
  return readFileSync(wasmPath);
}

function readCString(memory, addr) {
  const bytes = new Uint8Array(memory.buffer);
  let end = addr >>> 0;
  while (bytes[end] !== 0) {
    end++;
  }
  return new TextDecoder('ascii').decode(bytes.subarray(addr >>> 0, end));
}

function createImports(memory) {
  return {
    wasi_snapshot_preview1: {
      random_get(out, length) {
        const bytes = Buffer.alloc(length);
        randomFillSync(bytes);
        new Uint8Array(memory.buffer).set(bytes, out >>> 0);
        return 0;
      },
      clock_time_get(_clockId, _precision, out) {
        const view = new DataView(memory.buffer);
        const nowNs = BigInt(Date.now()) * 1_000_000n;
        view.setBigUint64(out >>> 0, nowNs, true);
        return 0;
      },
      proc_exit(code) {
        throw new Error(`WASM proc_exit(${code}) was called`);
      },
    },
    wasi: {
      'thread-spawn'() {
        throw new Error('Unexpected WASM thread spawn in single-threaded mont mul harness');
      },
    },
    env: {
      logstr(addr) {
        console.error(readCString(memory, addr));
      },
      throw_or_abort_impl(addr) {
        throw new Error(readCString(memory, addr));
      },
      env_hardware_concurrency() {
        return 1;
      },
      memory,
    },
  };
}

async function instantiateBarretenberg(bytes) {
  const module = await WebAssembly.compile(bytes);
  let lastError;

  for (const shared of [true, false]) {
    try {
      const memory = new WebAssembly.Memory({
        initial: DEFAULT_INITIAL_PAGES,
        maximum: DEFAULT_MAX_PAGES,
        shared,
      });
      const instance = await WebAssembly.instantiate(module, createImports(memory));
      return { instance, memory, shared };
    } catch (error) {
      lastError = error;
    }
  }

  throw lastError;
}

function createV8Helpers() {
  try {
    const prepare = new Function('fn', '%PrepareFunctionForOptimization(fn);');
    const optimizeOnNextCall = new Function('fn', '%OptimizeFunctionOnNextCall(fn);');
    const getStatus = new Function('fn', 'return %GetOptimizationStatus(fn);');
    return { prepare, optimizeOnNextCall, getStatus };
  } catch {
    return null;
  }
}

function warmUpRunner(runner, warmupIterations, v8Helpers) {
  if (v8Helpers) {
    v8Helpers.prepare(runner);
  }

  for (let i = 0; i < warmupIterations; i++) {
    runner();
  }

  if (!v8Helpers) {
    return null;
  }

  v8Helpers.optimizeOnNextCall(runner);
  runner();
  return v8Helpers.getStatus(runner);
}

function benchmarkRunner(runner, iterations, operationsPerCall) {
  const startedAt = process.hrtime.bigint();
  for (let i = 0; i < iterations; i++) {
    runner();
  }
  const elapsedNs = process.hrtime.bigint() - startedAt;
  const elapsedNsNumber = Number(elapsedNs);
  return {
    totalNs: elapsedNsNumber,
    totalMs: elapsedNsNumber / 1e6,
    averageNsPerCall: elapsedNsNumber / iterations,
    averageNsPerMul: elapsedNsNumber / (iterations * operationsPerCall),
  };
}

function formatNs(ns) {
  if (ns >= 1_000_000) {
    return `${(ns / 1_000_000).toFixed(2)} ms`;
  }
  if (ns >= 1_000) {
    return `${(ns / 1_000).toFixed(2)} us`;
  }
  return `${ns.toFixed(2)} ns`;
}

function readChecksum(memory, ptr) {
  const view = new DataView(memory.buffer, ptr, 16);
  const lo = view.getBigUint64(0, true).toString(16).padStart(16, '0');
  const hi = view.getBigUint64(8, true).toString(16).padStart(16, '0');
  return `0x${hi}${lo}`;
}

function selectFields(fieldSelection) {
  if (fieldSelection === 'all') {
    return Object.values(FIELD_CONFIGS);
  }

  const selected = FIELD_CONFIGS[fieldSelection];
  if (!selected) {
    throw new Error(`Unknown field "${fieldSelection}". Expected one of: fr, fq, all`);
  }
  return [selected];
}

function allocField(exports, fieldBytes) {
  return Number(exports.bbmalloc(fieldBytes));
}

function freeField(exports, ptr) {
  exports.bbfree(ptr);
}

function runFieldBenchmark(exports, memory, fieldConfig, fieldBytes, iterations, warmupIterations, v8Helpers) {
  if (typeof exports[fieldConfig.init] !== 'function' || typeof exports[fieldConfig.single] !== 'function') {
    console.log(`${fieldConfig.label}`);
    console.log('  required exports are missing; skipping');
    console.log('');
    return;
  }

  const lhs0Ptr = allocField(exports, fieldBytes);
  const rhs0Ptr = allocField(exports, fieldBytes);
  const lhs1Ptr = allocField(exports, fieldBytes);
  const rhs1Ptr = allocField(exports, fieldBytes);
  const out0Ptr = allocField(exports, fieldBytes);
  const out1Ptr = allocField(exports, fieldBytes);

  try {
    exports[fieldConfig.init](lhs0Ptr, rhs0Ptr, lhs1Ptr, rhs1Ptr);

    const singleExport = exports[fieldConfig.single];
    const singleRunner = () => {
      singleExport(lhs0Ptr, rhs0Ptr, out0Ptr);
    };

    const singleStatus = warmUpRunner(singleRunner, warmupIterations, v8Helpers);
    const singleMetrics = benchmarkRunner(singleRunner, iterations, 1);

    console.log(fieldConfig.label);
    console.log(
      `  single mont_mul: ${formatNs(singleMetrics.averageNsPerMul)}/mul average (${formatNs(
        singleMetrics.averageNsPerCall,
      )}/call, ${singleMetrics.totalMs.toFixed(2)} ms total, checksum ${readChecksum(memory, out0Ptr)})`,
    );

    if (singleStatus !== null) {
      console.log(`  single wrapper TF status (JS arrow, not wasm): ${singleStatus}`);
    }

    const pairedExport = exports[fieldConfig.paired];
    if (typeof pairedExport === 'function') {
      const pairedRunner = () => {
        pairedExport(lhs0Ptr, rhs0Ptr, lhs1Ptr, rhs1Ptr, out0Ptr, out1Ptr);
      };
      const pairedStatus = warmUpRunner(pairedRunner, warmupIterations, v8Helpers);
      const pairedMetrics = benchmarkRunner(pairedRunner, iterations, 2);
      console.log(
        `  paired mont_mul: ${formatNs(pairedMetrics.averageNsPerMul)}/mul average (${formatNs(
          pairedMetrics.averageNsPerCall,
        )}/call, ${pairedMetrics.totalMs.toFixed(2)} ms total, checksum ${readChecksum(memory, out0Ptr)} / ${readChecksum(
          memory,
          out1Ptr,
        )})`,
      );
      if (pairedStatus !== null) {
        console.log(`  paired wrapper TF status (JS arrow, not wasm): ${pairedStatus}`);
      }
    } else {
      console.log('  paired mont_mul: export not present in this wasm build; skipped');
    }

    console.log('');
  } finally {
    freeField(exports, lhs0Ptr);
    freeField(exports, rhs0Ptr);
    freeField(exports, lhs1Ptr);
    freeField(exports, rhs1Ptr);
    freeField(exports, out0Ptr);
    freeField(exports, out1Ptr);
  }
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.rebuild) {
    rebuildCanonicalWasm();
  }
  const wasmPath = resolveWasmPath(options.wasmPath);
  const bytes = loadWasmBytes(wasmPath);
  const { instance, memory, shared } = await instantiateBarretenberg(bytes);
  const exports = instance.exports;

  if (typeof exports._initialize !== 'function') {
    throw new Error('Loaded wasm does not export _initialize');
  }
  if (typeof exports.bbmalloc !== 'function' || typeof exports.bbfree !== 'function') {
    throw new Error('Loaded wasm does not export bbmalloc/bbfree');
  }
  if (typeof exports.wasm_bench_field_element_size_bytes !== 'function') {
    throw new Error('Loaded wasm does not export the mont mul bench entry points');
  }

  exports._initialize();

  const fieldBytes = Number(exports.wasm_bench_field_element_size_bytes());
  const v8Helpers = createV8Helpers();

  console.log(`WASM: ${wasmPath}${options.rebuild ? ' (freshly rebuilt)' : ' (rebuild skipped)'}`);
  console.log(`Memory import: ${shared ? 'shared' : 'unshared'}`);
  console.log(`TurboFan natives: ${v8Helpers ? 'enabled' : 'unavailable (run with --allow-natives-syntax to force optimization)'}`);
  console.log(`Iterations: ${options.iterations}`);
  console.log(`Warmup: ${options.warmup}`);
  console.log('');

  for (const fieldConfig of selectFields(options.field)) {
    runFieldBenchmark(exports, memory, fieldConfig, fieldBytes, options.iterations, options.warmup, v8Helpers);
  }
}

await main().catch(error => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
