# PG2024-Data-Parallel-Ray-Tracing

The main process is in the render.cpp file.

Note that this project file cannot be compiled and run.

```
1. Initialize CUDA, OptiX, and MPI.

2. for each sample:
        for each bounce:
            Set up OptiX parameters and copy them to the device

            if(bounce > 0)
                Cast secondary rays with nerual proxies

            Repeat until all paths are completed:
                Launch the main ray tracing pipeline
                Perform scanning and reordering for results

                Perform MPI all-to-all communication to transfer path data

                if all paths are completed:
                    break

            Generate shadow & secondary ray

            Launch shadow ray tracing pipeline

            Cast shadow rays with nerual proxies

    Perform MPI reduce operation to aggregate image data across all nodes

3.Save the final image.