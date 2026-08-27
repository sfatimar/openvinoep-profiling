# Profiling and Debugging: OpenVINO EP vs. CPU (MLAS) EP

A field-level comparison of what the ONNX Runtime profiler reports when a graph runs on the
OpenVINO plugin EP versus the built-in CPU (MLAS) EP.

**OpenVINO plugin EP** 
Profiling can be enabled through: 
1. You can enable ONNX Runtime latency profiling in code:
''
import onnxruntime as rt

sess_options = rt.SessionOptions()
sess_options.enable_profiling = True
''
2. The onnxruntime_perf_test.exe tool (available from the build drop) can be used to test various knobs. 
Please find the usage instructions using onnxruntime_perf_test.exe -h. Profiling can be enabled through -p option
The [perf_view tool](https://github.com/microsoft/onnxruntime/tree/main/tools/perf_view) can also be used to render the statistics as a summarized view in the browser.

The JSON file which contains the detailed performance data (latency of each operator, etc). 
This file is a standard performance tracing file, and to view it in a user-friendly way, you can open it by using multiple tools.

(Windows) Use the WPA GUI to open the trace using the Perfetto OSS plugin - Microsoft-Performance-Tools-Linux-Android
Perfetto UI - Successor to Chrome Tracing UI
chrome://tracing:
Open a Chromium based browser such as Edge or Chrome
Type chrome://tracing in the address bar
Load the generated JSON file

```bash
onnxruntime_perf_test.exe \
  --plugin_ep_libs "OpenVINOExecutionProvider|onnxruntime_providers_openvino_plugin.dll" \
  --plugin_eps OpenVINOExecutionProvider \
  --filter_ep_devices "ov_device|CPU" \
  -I -m times -r 1 \
  --plugin_ep_options "load_config|load_config.json" \
  -p ov_profile \
  <model.onnx>
```

`load_config.json` — root keys are OpenVINO device names, values are property maps:

```json
{
  "CPU": {
    "PERF_COUNT": "YES"
  }
}
```


**CPU / MLAS EP** baseline:

```bash
onnxruntime_perf_test.exe -e cpu -I -m times -r 1 -p mlas_profile <model.onnx>
```

OVEP fuses its whole partition into **one** ORT node, so ORT's own instrumentation — type/shape,
memory, thread scheduling — fires exactly once, describing the subgraph as a black box. Per-op
detail arrives through a *separate and much thinner* channel: the `Kernel` events fed from
`ov::InferRequest::get_profiling_info()`.

## 3. Field matrix

`Y` = present and populated · `Z` = present but zero-filled / sentinel · `—` = absent

| Field | MLAS `Node` | OVEP fused `Node` | OVEP `Kernel` | Notes |
|---|:--:|:--:|:--:|---|
| `op_name` | Y | Y | — | OVEP node reports the fused-partition name |
| `node_index` | Y | Y | — | |
| `provider` | Y | Y | — | `Kernel` events use `ep` instead |
| `input_type_shape` / `output_type_shape` | Y | Y | **—** | boundary-only for OVEP |
| `parameter_size` | Y | **Z** (always 0) | — | weights live in the OV blob, invisible to ORT |
| `activation_size` / `output_size` | Y | Y | — | OVEP: subgraph I/O only |
| `mem_bytes_in_use` | Y | Y | — | OVEP: ORT-side I/O arena only |
| `mem_bytes_requested_in_use` | Y | Y | — | |
| `mem_in_use_peak` | Y | Y | — | |
| `mem_arena_held` | Y | Y | — | |
| `mem_in_use_delta` | Y | Y | — | first inference only (see §5) |
| `mem_requested_in_use_delta` | Y | Y | — | first inference only |
| `mem_arena_held_delta` | Y | Y | — | first inference only |
| `thread_scheduling_stats.main_thread.core` | Y | **Z** (`-1`) | — | **see §4** |
| `thread_scheduling_stats.main_thread.block_size` | Y | **Z** (`[]`) | — | |
| `…main_thread.{Distribution,DistributionEnqueue,Run,Wait,WaitRevoke}` | Y | **Z** (all `0`) | — | |
| `thread_scheduling_stats.sub_threads[*].core` | Y | **Z** (all `-1`) | — | |
| `thread_scheduling_stats.sub_threads[*].num_run` | Y | **Z** (all `0`) | — | |
| `ov_exec_type` | — | — | **Y** | **OVEP-only, see §6** |
| `ov_node_type` | — | — | **Y** | OpenVINO internal op type |
| `ov_status` | — | — | Y | filtered to `EXECUTED` only |
| `parent_ort_event_name` / `ort_correlation_id` | — | — | Y | links `Kernel` → fused `Node` |

The `Z` rows are the dangerous ones. The keys are present with plausible structure, so tooling
that reads the JSON sees a populated schema and silently reports "core 0 usage" or "2 MB peak"
as if measured.

---

## 4. Thread scheduling, core affinity and Memory Profiling — the largest gap

**This is the category with no OVEP substitute at all.**

### Why

These counters are instrumented inside **ORT's own intra-op thread pool**. OVEP hands the
subgraph to OpenVINO, which schedules on its **own TBB pool** (`tbb12.dll`, `tbbbind_2_5.dll`).
ORT's calling thread simply blocks for the duration, so ORT has nothing to count. The plugin's
profiling path does not collect TBB-side thread or affinity data.

## 5. Operator Report


- **`ov_exec_type` — the selected kernel implementation.** It names the actual microkernel chosen (ISA and precision, e.g.
  `brgconv_avx2_f32`, `jit_avx2_f32`, `jit_uni_f16`). 
- **`ov_status` — graph-compiler disposition** (`EXECUTED` / `NOT_RUN` / `OPTIMIZED_OUT`).
  In practice the plugin filters to `EXECUTED` before emitting (`plugin/ov_ep_profiler.cc:262`),
  so the field is always `"EXECUTED"` in the report and ops OpenVINO folded away simply do not
  appear. Absence, not a status value, is the signal that an op was optimised out.
- ** `ov_node_type` - exposes OpenVINO's internal op taxonomy, which reveals fusions and
internal reorders that have no ONNX-level equivalent.
    ### Cross-EP op comparison is not a name join

MLAS `op_name` values are ONNX op types plus ORT-inserted layout ops; OVEP `ov_node_type` values
are OpenVINO internal types. Semantically equivalent operations carry different names across the
two profiles. Comparing per-op cost between EPs requires an explicit mapping table — it is not a
join on op name.
- ** OVEP `Kernel` durations can exceed their parent `Node` duration.** They are OpenVINO
   `real_time` counters captured independently and may overlap across OpenVINO's own threads.
   The upstream API documents this: *"Due to parallel execution, the total execution time for all
   nodes might be greater than the total inference time.
- **Tiled operations report a sum, not wall-clock.** Where OpenVINO executes an op using tiling,
   the reported time is the sum across tiles. Those durations are not elapsed time.
- **OVEP `Kernel` `pid` / `tid` are synthetic.** `pid` is `-1` and `tid` values are lane numbers
   handed out by `OverlapAwareTidAllocator` (`plugin/ov_ep_profiler.cc:207`) purely so overlapping
   ops render on separate rows in a trace viewer. **They are not thread IDs.** Reading them as
   threads will produce a completely fictional concurrency picture. MLAS `tid` values *are* real
   OS thread IDs.


