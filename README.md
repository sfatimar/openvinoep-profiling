# openvinoep-profiling

This repository is created to provide built-in scripts and documentation for OpenVINO Execution Provider Profiling. 

Tools:
1. [WinMLCLI Tools](https://github.com/microsoft/winml-cli)
''
uv run winml perf -m resnet_out/model.onnx --device auto --iterations 50 --monitor
''

2. onnxruntime_perf_test.exe This tool can be found in WinML EP Test Tools/Profiling Packages can be found at [WINML EP artifactory](https://gfx-assets-build.intel.com/artifactory/onnxruntime-builds/ci/)

3. benchmark_app.exe: OpenVINO Benchmarking tool can be found with individual OpenVINO Release [Artifactory](https://storage.openvinotoolkit.org/repositories/)

Scripts:

Examples:
1. Resnet
1. Bert

Profiling: 
  1. Latency Profiling
     For Operator Profiling please go through LatencyProfiling.md
  2. Power and Performance Measurement through VTune Profiler
  3. Memory Profiling 
  3. OpenVINO Detailed Profiling Tools
  
   
