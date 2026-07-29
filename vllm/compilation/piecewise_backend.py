# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import dataclasses
import io
import json
import pickle
from collections.abc import Callable
from pickle import Pickler
from typing import Any

import torch._functorch.config
import torch.fx as fx
from torch._dynamo.utils import dynamo_timed
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from torch._logging._internal import trace_structured
from torch.fx._graph_pickler import GraphPickler, Options


from vllm.compilation.backends import VllmBackend
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.logger import init_logger

logger = init_logger(__name__)


def find_raw_node_path(obj: Any, path: str = "", memo: Any = None) -> list[str]:
    if memo is None:
        memo = set()
    if id(obj) in memo:
        return []
    memo.add(id(obj))

    import torch
    if isinstance(obj, torch.fx.Node):
        return [f"{path} (Node: {obj.name})"]

    paths = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            paths.extend(find_raw_node_path(v, f"{path}.{k}", memo))
    elif isinstance(obj, (list, tuple, set)):
        for i, v in enumerate(obj):
            paths.extend(find_raw_node_path(v, f"{path}[{i}]", memo))
    elif isinstance(obj, slice):
        paths.extend(find_raw_node_path(obj.start, f"{path}.start", memo))
        paths.extend(find_raw_node_path(obj.stop, f"{path}.stop", memo))
        paths.extend(find_raw_node_path(obj.step, f"{path}.step", memo))
    elif dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            paths.extend(find_raw_node_path(getattr(obj, f.name), f"{path}.{f.name}", memo))
    elif hasattr(obj, "__dict__"):
        paths.extend(find_raw_node_path(obj.__dict__, path, memo))

    return paths


def log_raw_node_leakage(graph: fx.GraphModule):
    logger.info("Inspecting graph for raw Node leakage in metadata...")
    found_any = False
    for node in graph.graph.nodes:
        for k, v in node.meta.items():
            paths = find_raw_node_path(v, f"node({node.name}).meta.{k}")
            if paths:
                found_any = True
                for p in paths:
                    logger.warning("💥 Leakage detected: %s", p)
    if not found_any:
        logger.info("No raw Node leakage detected in graph metadata.")


def get_fake_args_from_graph(graph: fx.GraphModule) -> list[Any]:
    """Get fake args directly from graph placeholder nodes."""
    fake_args = []
    for node in graph.graph.nodes:
        if node.op == "placeholder":
            fake_args.append(node.meta["example_value"])
        else:
            break
    return fake_args


def create_concrete_args(graph: fx.GraphModule, size: int) -> list[Any]:
    """Create Fake example inputs with symbolic dims replaced by a concrete size.

    Used for single-size compilation where we need concrete-shaped inputs.
    The Dynamo-captured graph gives us example inputs with SymInts in them.
    """
    from torch._prims_common import compute_required_storage_length
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv, is_symbolic

    def concretize(sym_val: Any) -> int:
        """Replace all symbolic variables in a SymInt expression with size."""
        if not is_symbolic(sym_val):
            return int(sym_val)
        expr = sym_val.node.expr
        return int(expr.subs({s: size for s in expr.free_symbols}))

    fake_mode = FakeTensorMode(shape_env=ShapeEnv())

    args: list[Any] = []
    with fake_mode:
        for node in graph.graph.nodes:
            if node.op != "placeholder":
                break
            val = node.meta["example_value"]
            if isinstance(val, torch.SymInt):
                args.append(concretize(val))
            elif isinstance(val, torch.Tensor):
                new_shape = tuple(concretize(d) for d in val.shape)
                new_strides = tuple(concretize(s) for s in val.stride())
                new_storage_offset = concretize(val.storage_offset())
                needed_size = compute_required_storage_length(
                    new_shape, new_strides, new_storage_offset
                )
                t = torch.empty(needed_size, dtype=val.dtype, device=val.device)
                t = t.as_strided(new_shape, new_strides, new_storage_offset)
                args.append(t)
            else:
                args.append(val)
    return args


def _compile_range_helper_clean(
    compiler_manager: Any,
    graph_bytes: bytes,
    args_list_bytes: bytes,
    inductor_config: dict[str, Any],
    compilation_config: Any,
    compile_range: Any,
    graph_index: int,
    num_graphs: int,
    is_encoder: bool,
) -> tuple[Any, Any]:
    """Helper function to compile a single range in a separate process with GraphPickler."""
    import sys
    sys.stderr.write(f"[CHILD PROCESS] Starting clean compile helper for range {compile_range}\n")
    sys.stderr.flush()
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv
        from torch.fx._graph_pickler import GraphPickler

        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        with fake_mode:
            logger.info("[CHILD PROCESS] Deserializing graph and args_list...")
            graph = GraphPickler.loads(graph_bytes, fake_mode)
            args_list = GraphPickler.loads(args_list_bytes, fake_mode)

            logger.info(
                "[Before compile] Child process compile range %s",
                compile_range
            )
            compiler_manager.compile(
                graph,
                args_list,
                inductor_config,
                compilation_config,
                compile_range=compile_range,
                graph_index=graph_index,
                num_graphs=num_graphs,
                is_encoder=is_encoder,
            )
            logger.info(
                "[After compile] Child process compile range %s",
                compile_range
            )
        cache_key = (compile_range, graph_index, compiler_manager.compiler.name)
        cache_value = compiler_manager.cache.get(cache_key)
        sys.stderr.write(f"[CHILD PROCESS] Finished compile helper for range {compile_range}, cache_key={cache_key}, cache_value={cache_value}\n")
        sys.stderr.flush()
        return cache_key, cache_value
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        sys.stderr.write(f"[CHILD PROCESS] Exception in compile helper for range {compile_range}: {tb}\n")
        sys.stderr.flush()
        logger.error("Compilation failed in child process: %s", tb)
        raise RuntimeError(f"Error compiling range {compile_range}:\n{tb}")


def _unpickle_weak_value_dict(d_dict):
    import weakref
    wvd = weakref.WeakValueDictionary()
    for k, v in d_dict.items():
        wvd[k] = v
    return wvd


def _unpickle_as_weakref(referent):
    import weakref
    return weakref.ref(referent)


def _unpickle_as_dead_weakref():
    return lambda: None


def _clear_fake_mode_recursive(metadata: Any) -> Any:
    if metadata is None:
        return None

    from torch._subclasses.meta_utils import MetaTensorDesc
    if not isinstance(metadata, MetaTensorDesc):
        return metadata

    updates = {}
    if metadata.fake_mode is not None:
        updates["fake_mode"] = None

    # List of fields that are MetaTensorDesc
    fields_to_clear = [
        "base", "grad", "unwrapped", "values",
        "crow_indices", "col_indices", "ccol_indices", "row_indices"
    ]
    for field in fields_to_clear:
        val = getattr(metadata, field, None)
        if val is not None:
            updates[field] = _clear_fake_mode_recursive(val)

    # Handle attrs (dict of MetaTensorDesc)
    if metadata.attrs is not None:
        new_attrs = {}
        for k, v in metadata.attrs.items():
            new_attrs[k] = _clear_fake_mode_recursive(v)
        updates["attrs"] = new_attrs

    if updates:
        import dataclasses
        return dataclasses.replace(metadata, **updates)
    return metadata


class PiecewiseGraphPickler(GraphPickler):
    def reducer_override(self, obj: object) -> Any:
        import weakref
        from torch._subclasses.fake_tensor import FakeTensor
        from torch.fx._graph_pickler import _TensorPickleData

        if isinstance(obj, FakeTensor):
            res = _TensorPickleData.reduce_helper(self, obj)
            if isinstance(res, tuple) and len(res) == 2:
                reduce_fn, args = res
                if isinstance(args, tuple) and len(args) == 2:
                    tensor_pickle_data, unpickle_stateToken = args
                    if isinstance(tensor_pickle_data, _TensorPickleData):
                        tensor_pickle_data.metadata = _clear_fake_mode_recursive(tensor_pickle_data.metadata)
            return res

        if isinstance(obj, weakref.WeakValueDictionary):
            return (_unpickle_weak_value_dict, (dict(obj),))

        if isinstance(obj, weakref.ref):
            referent = obj()
            if referent is not None:
                return (_unpickle_as_weakref, (referent,))
            else:
                return (_unpickle_as_dead_weakref, ())

        return super().reducer_override(obj)


@dataclasses.dataclass
class RangeEntry:
    compile_range: Range
    compiled: bool = False
    runnable: Callable[..., Any] = None  # type: ignore


class PiecewiseBackend:
    def __init__(
        self,
        graph: fx.GraphModule | None,
        vllm_config: VllmConfig,
        piecewise_compile_index: int,
        total_piecewise_compiles: int,
        sym_shape_indices: list[int],
        vllm_backend: VllmBackend,
        returns_tuple: bool,
        compiled_runnables: dict[str, Callable[..., Any]] | None = None,
        submod_name: str = "",
    ):
        """
        The backend for piecewise compilation.
        It mainly handles the compilation of static shapes and
        dispatching based on runtime shape.

        We will compile `self.graph` once for the general shape,
        and then compile for different shapes specified in
        `compilation_config.compile_sizes`.

        This class supports two mutually exclusive modes:
        1. Compilation (graph is set, compiled_runnables is None):
           Used during initial compilation when we have the FX graph
           and need to compile it for each shape range.
        2. Precompilation (graph is None, compiled_runnables is set):
           Used when loading from cache/AOT artifacts where we already
           have pre-compiled callables and don't need the original graph.

        Exactly one of graph or compiled_runnables must be provided.
        """
        assert bool(graph is not None) ^ bool(compiled_runnables is not None), (
            "exactly one of graph and compiled_runnables should be set."
        )

        self.graph = graph
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.piecewise_compile_index = piecewise_compile_index
        self.total_piecewise_compiles = total_piecewise_compiles
        self.vllm_backend = vllm_backend
        self.compiled_runnables = compiled_runnables
        self.submod_name = submod_name

        self.is_first_graph = piecewise_compile_index == 0
        self.is_last_graph = piecewise_compile_index == total_piecewise_compiles - 1

        self.is_full_graph = total_piecewise_compiles == 1
        self.is_encoder_compilation = vllm_backend.is_encoder

        self.compile_ranges = self.compilation_config.get_compile_ranges()
        if self.is_encoder_compilation:
            # For encoder compilation we use the max int32 value
            # to set the upper bound of the compile ranges
            max_int32 = 2**31 - 1
            last_compile_range = self.compile_ranges[-1]
            assert (
                last_compile_range.end
                == vllm_config.scheduler_config.max_num_batched_tokens
            )
            self.compile_ranges[-1] = Range(
                start=last_compile_range.start, end=max_int32
            )

        log_string = f"PiecewiseBackend: compile_ranges: {self.compile_ranges}"
        logger.debug_once(log_string)

        self.compile_sizes = self.compilation_config.compile_sizes
        log_string = f"PiecewiseBackend: compile_sizes: {self.compile_sizes}"
        logger.debug_once(log_string)

        self.sym_shape_indices = sym_shape_indices
        self.returns_tuple = returns_tuple

        # the entries for ranges that we need to either
        self.range_entries: dict[Range, RangeEntry] = {}

        # We only keep compilation management inside this class directly.
        if self.compile_sizes is not None:
            for size in self.compile_sizes:
                if isinstance(size, str):
                    assert size == "cudagraph_capture_sizes"
                    raise NotImplementedError(
                        "cudagraph_capture_sizes not supported in compile_sizes."
                        "This should be handled in `post_init_cudagraph_sizes`."
                    )
                else:
                    assert isinstance(size, int)
                    range = Range(start=size, end=size)
                    if range not in self.compile_ranges:
                        self.range_entries[range] = RangeEntry(
                            compile_range=range,
                        )

        for range in self.compile_ranges:
            self.range_entries[range] = RangeEntry(
                compile_range=range,
            )

        # Track whether we've logged the graph for this subgraph (only log once)
        self._graph_logged = False

        if self.graph is not None:
            self.compile_all_ranges()
        else:
            self.load_all_ranges()

    def get_compiled_graph_wrapper(
        self, compiled_graph: Callable[..., Any]
    ) -> Callable[..., Any]:
        def compiled_graph_wrapper(*args: Any) -> Any:
            graph_output = compiled_graph(*args)
            # unpack the tuple if needed
            # TODO(rzou): the implication is that we're not
            # reading the python bytecode correctly in vLLM?
            if self.returns_tuple or not isinstance(graph_output, (tuple, list)):
                return graph_output
            else:
                return graph_output[0]

        return compiled_graph_wrapper

    def to_bytes(self) -> dict[str, bytes]:
        class StandaloneCompiledArtifactsPickler(Pickler):
            def reducer_override(self, obj: object) -> Any:
                if isinstance(obj, CachingAutotuner):
                    obj.prepare_for_pickle()
                    return pickle.loads, (
                        pickle.dumps(
                            obj,
                        ),
                    )
                return NotImplemented

        def serialize(fn: Callable[..., Any]) -> bytes:
            assert hasattr(fn, "serialize"), "fn must have serialize method"
            with torch._functorch.config.patch("bundled_autograd_cache", True):
                entry = fn.serialize()

                f = io.BytesIO()
                StandaloneCompiledArtifactsPickler(f).dump(entry)
                result = f.getvalue()
            return result

        out = {}

        for range_key, entry in self.range_entries.items():
            if not entry.compiled:
                logger.debug(
                    "entry with range %s not compiled, so cannot get its bytes",
                    range_key,
                )
                continue
            if hasattr(entry.runnable, "serialize"):
                out[str(range_key)] = serialize(entry.runnable)

        return out

    def _check_graph_picklability(self) -> None:
        import pickle

        assert self.graph is not None

        logger.info("Checking picklability of the graph module...")
        try:
            pickle.dumps(self.graph)
            logger.info("GraphModule is fully picklable.")
            return
        except Exception as e:
            logger.warning(
                "GraphModule failed to pickle: %s. Analyzing components...", e
            )

        # Check individual nodes and their meta
        for node in self.graph.graph.nodes:
            try:
                pickle.dumps(node.op)
                pickle.dumps(node.name)
                pickle.dumps(node.args)
                pickle.dumps(node.kwargs)
            except Exception as e:
                logger.warning(
                    "Node %s basic attributes are not picklable: %s", node.name, e
                )

            for k, v in node.meta.items():
                try:
                    pickle.dumps(v)
                except Exception as e:
                    logger.warning(
                        "Node %s metadata key '%s' (type: %s) is not picklable: %s",
                        node.name,
                        k,
                        type(v),
                        e,
                    )
    
    def _check_arguments_picklability_clean(
        self,
        compiler_manager: Any,
        args_list: list[Any],
        compilation_config: Any,
        compile_range: Range,
    ) -> None:
        import pickle
        import torch
        logger.info("Checking picklability of clean arguments for range %s", compile_range)
        for name, val in [
            ("clean_compiler_manager", compiler_manager),
            ("graph", self.graph),
            ("args_list", args_list),
            ("inductor_config", self.vllm_backend.inductor_config),
            ("clean_compilation_config", compilation_config),
        ]:
            if name == "clean_compiler_manager":
                logger.info("Clean CompilerManager fields:")
                visited = set()
                def find_tensors(obj, path=""):
                    import torch
                    if isinstance(obj, torch.Tensor):
                        logger.info("FOUND TENSOR at %s: device=%s, shape=%s", path, obj.device, obj.shape)
                        return
                    if id(obj) in visited:
                        return
                    visited.add(id(obj))
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            find_tensors(v, f"{path}[{k!r}]")
                    elif isinstance(obj, (list, tuple, set)):
                        for i, v in enumerate(obj):
                            find_tensors(v, f"{path}[{i}]")
                    elif hasattr(obj, "__dict__"):
                        for k, v in obj.__dict__.items():
                            find_tensors(v, f"{path}.{k}")
                find_tensors(val, "clean_compiler_manager")
            try:
                logger.info("Checking picklability of %s", name)
                pickle.dumps(val)
                logger.info("Argument %s is picklable", name)
            except Exception as pe:
                logger.error("Argument %s is NOT picklable: %s", name, pe)
                if name == "args_list":
                    for i, arg in enumerate(val):
                        try:
                            pickle.dumps(arg)
                        except Exception as ae:
                            logger.error("args_list[%d] (type: %s) failed to pickle: %s", i, type(arg), ae)
                            if isinstance(arg, torch.Tensor):
                                logger.error("Tensor device: %s, dtype: %s, shape: %s", arg.device, arg.dtype, arg.shape)

    def _compile_all_ranges_sequential(
        self, ranges_to_compile: list[RangeEntry]
    ) -> None:
        for range_entry in ranges_to_compile:
            self._log_compile_start(range_entry.compile_range)

            if range_entry.compile_range.is_single_size():
                args_list = create_concrete_args(
                    self.graph, range_entry.compile_range.start
                )
            else:
                args_list = get_fake_args_from_graph(self.graph)

            range_entry.runnable = self.vllm_backend.compiler_manager.compile(
                self.graph,
                args_list,
                self.vllm_backend.inductor_config,
                self.compilation_config,
                compile_range=range_entry.compile_range,
                graph_index=self.piecewise_compile_index,
                num_graphs=self.total_piecewise_compiles,
                is_encoder=self.vllm_backend.is_encoder,
            )

            range_entry.compiled = True

    def _compile_all_ranges_parallel_clean(
        self, ranges_to_compile: list[RangeEntry]
    ) -> None:
        import multiprocessing
        import copy
        from concurrent.futures import ProcessPoolExecutor
        from vllm.compilation.backends import CompilerManager
        from torch.fx._graph_pickler import GraphPickler, Options

        # Run picklability check and log results before starting pool
        self._check_graph_picklability()

        # Create a clean compilation config copy
        clean_compilation_config = copy.copy(self.compilation_config)
        # Clear the static forward context which contains TPU tensors (weights, caches)
        if hasattr(clean_compilation_config, "static_forward_context"):
            clean_compilation_config.static_forward_context = {}

        # Create a clean compiler manager copy for workers to avoid pickling TPU state
        clean_compiler_manager = CompilerManager(clean_compilation_config)
        clean_compiler_manager.compiler.cache_dir = self.vllm_backend.compiler_manager.compiler.cache_dir
        clean_compiler_manager.compiler._disable_cache = self.vllm_backend.compiler_manager.compiler._disable_cache
        clean_compiler_manager.disable_cache = self.vllm_backend.compiler_manager.disable_cache
        clean_compiler_manager.cache_file_path = self.vllm_backend.compiler_manager.cache_file_path

        # Determine number of workers
        max_workers = min(os.cpu_count() or 1, len(ranges_to_compile))
        env_max_workers = os.getenv("VLLM_PARALLEL_COMPILE_MAX_WORKERS")
        if env_max_workers is not None:
            max_workers = min(max_workers, int(env_max_workers))
        else:
            max_workers = min(max_workers, 4)  # Default to 4

        logger.info(
            "Compiling %d ranges in parallel using %d processes (with clean CompilerManager)",
            len(ranges_to_compile),
            max_workers,
        )

        mp_ctx = multiprocessing.get_context("spawn")

        # Workaround for "daemonic processes are not allowed to have children"
        # when running inside vLLM worker processes (which are spawned as daemon).
        curr_proc = multiprocessing.current_process()
        is_daemon = curr_proc.daemon
        if is_daemon:
            logger.info(
                "Temporarily disabling daemon flag on current process (%s) "
                "to allow spawning parallel compilation workers.",
                curr_proc.name,
            )
            curr_proc.daemon = False

        try:
            with ProcessPoolExecutor(
                max_workers=max_workers, mp_context=mp_ctx
            ) as executor:
                futures = []
                for entry in ranges_to_compile:
                    self._log_compile_start(entry.compile_range)

                    if entry.compile_range.is_single_size():
                        args_list = create_concrete_args(
                            self.graph, entry.compile_range.start
                        )
                    else:
                        args_list = get_fake_args_from_graph(self.graph)
                    
                    self._check_arguments_picklability_clean(
                        clean_compiler_manager,
                        args_list,
                        clean_compilation_config,
                        entry.compile_range
                    )

                    logger.info(
                        "Parent before submitting compilation for range (%s)",
                        entry.compile_range
                    )

                    log_raw_node_leakage(self.graph)
                    options = Options(ops_filter=None)
                    graph_bytes = PiecewiseGraphPickler.dumps(self.graph, options=options)
                    args_list_bytes = PiecewiseGraphPickler.dumps(args_list, options=options)

                    future = executor.submit(
                        _compile_range_helper_clean,
                        clean_compiler_manager,
                        graph_bytes,
                        args_list_bytes,
                        self.vllm_backend.inductor_config,
                        clean_compilation_config,
                        entry.compile_range,
                        self.piecewise_compile_index,
                        self.total_piecewise_compiles,
                        self.vllm_backend.is_encoder,
                    )

                    logger.info(
                        "Parent after submitting compilation for range (%s)",
                        entry.compile_range
                    )

                    futures.append((entry, future))

                for entry, future in futures:
                    cache_key, cache_value = future.result()

                    # Update cache in main process
                    if cache_key is not None and cache_value is not None:
                        self.vllm_backend.compiler_manager.cache[cache_key] = cache_value

                    # Re-generate args_list for loading in main process
                    if entry.compile_range.is_single_size():
                        args_list = create_concrete_args(
                            self.graph, entry.compile_range.start
                        )
                    else:
                        args_list = get_fake_args_from_graph(self.graph)

                    # Load the compiled graph in main process
                    compiled_graph = self.vllm_backend.compiler_manager.load(
                        self.graph,
                        args_list,
                        self.piecewise_compile_index,
                        entry.compile_range,
                    )

                    assert compiled_graph is not None, (
                        f"Failed to load compiled graph for range {entry.compile_range}"
                    )

                    entry.runnable = compiled_graph
                    entry.compiled = True

                if len(futures) > 0:
                    self.vllm_backend.compiler_manager.is_cache_updated = True
        finally:
            if is_daemon:
                curr_proc.daemon = True

    def compile_all_ranges(self) -> None:
        """Compile all range entries for this piecewise subgraph up front."""
        assert self.graph is not None, (
            "Cannot compile without a graph. "
            "When loading from cache/AOT artifacts, "
            "compile_all_ranges should not be called."
        )

        ranges_to_compile = [
            entry for entry in self.range_entries.values() if not entry.compiled
        ]

        if not ranges_to_compile:
            return

        disable_cache = self.vllm_backend.compiler_manager.disable_cache

        # Read configuration directly from environment
        disable_parallel = bool(
            int(os.getenv("VLLM_DISABLE_PARALLEL_COMPILE", "0"))
        )

        reasons = []
        if len(ranges_to_compile) <= 1:
            reasons.append(f"only {len(ranges_to_compile)} range to compile")
        if disable_cache:
            reasons.append("compilation cache is disabled")
        if disable_parallel:
            reasons.append("parallel compilation is explicitly disabled")

        use_parallel = not reasons

        if use_parallel:
            logger.info("Performing parallel piecewise compilation.")
            # If parallel compilation fails, it will raise an error and fail fast.
            self._compile_all_ranges_parallel_clean(ranges_to_compile)
        else:
            logger.info(
                "Performing sequential piecewise compilation. Reasons: %s",
                ", ".join(reasons),
            )
            self._compile_all_ranges_sequential(ranges_to_compile)

    @dynamo_timed("vllm_log_compile_start_torch_trace_only")
    def _log_compile_start(self, compile_range: Range):
        """Log compilation event for TORCH_TRACE/tlparse."""
        is_cudagraph_size = (
            self.compile_sizes is not None and compile_range.start in self.compile_sizes
        )
        subgraph_index = self.piecewise_compile_index
        submod_name = self.submod_name
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "vllm_piecewise_compile_start",
                "encoding": "json",
            },
            payload_fn=lambda: json.dumps(
                {
                    "piecewise_index": subgraph_index,
                    "submod_name": submod_name,
                    "total_piecewise_compiles": self.total_piecewise_compiles,
                    "compile_range_start": compile_range.start,
                    "compile_range_end": compile_range.end,
                    "is_single_size": compile_range.is_single_size(),
                    "is_cudagraph_capture_size": is_cudagraph_size,
                }
            ),
        )

        # Log the subgraph graph dump only once per subgraph (not per size)
        # to reduce log file size. The graph code is the same for all sizes.
        if not self._graph_logged:
            self._graph_logged = True
            assert self.graph is not None
            trace_structured(
                "graph_dump",
                metadata_fn=lambda: {
                    "name": f"vllm_{submod_name}",
                },
                payload_fn=lambda: self.graph.print_readable(print_output=False),
            )

    def load_all_ranges(self) -> None:
        """Load all pre-compiled runnables for this piecewise subgraph.

        Called during warm start to wrap all cached compiled_runnables
        into range_entry.runnable up front, analogous to compile_all_ranges()
        for the cold start path.
        """
        assert self.compiled_runnables is not None, (
            "load_all_ranges should only be called when compiled_runnables "
            "is set (warm start / cache loading path)."
        )
        for range_entry in self.range_entries.values():
            if range_entry.compiled:
                continue
            key = str(range_entry.compile_range)
            assert key in self.compiled_runnables, (
                f"Missing compiled runnable for range {range_entry.compile_range}. "
                f"Available keys: {list(self.compiled_runnables.keys())}"
            )
            range_entry.runnable = self.get_compiled_graph_wrapper(
                self.compiled_runnables[key]
            )
            range_entry.compiled = True

    def _find_range_for_shape(self, runtime_shape: int) -> RangeEntry | None:
        # First we try to find the range entry for the concrete compile size
        # If not found, we search for the range entry
        # that contains the runtime shape.
        if self.compile_sizes is None:
            return None

        if runtime_shape in self.compile_sizes:
            return self.range_entries[Range(start=runtime_shape, end=runtime_shape)]
        else:
            for range in self.compile_ranges:
                if runtime_shape in range:
                    return self.range_entries[range]
        return None

    def __call__(self, *args: Any) -> Any:
        if self.sym_shape_indices:
            runtime_shape = args[self.sym_shape_indices[0]]
            range_entry = self._find_range_for_shape(runtime_shape)
            assert range_entry is not None, (
                f"Shape: {runtime_shape} out of considered ranges: "
                f"{self.compile_ranges}"
            )
        else:
            # All inputs have static shapes; use the only compiled range_entry
            compiled_entries = [re for re in self.range_entries.values() if re.compiled]
            assert len(compiled_entries) == 1, (
                f"Expected exactly one compiled range_entry for static shape "
                f"compilation, but found {len(compiled_entries)}"
            )
            range_entry = compiled_entries[0]

        assert range_entry.compiled, (
            "All ranges should be compiled or loaded up front in "
            "PiecewiseBackend.__init__. "
            f"range_entry={range_entry.compile_range}"
        )
        return range_entry.runnable(*args)
