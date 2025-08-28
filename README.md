# FLASH Parallel I/O Optimization Demo

This repository contains a demonstration of an Artificial Neural Network (ANN)-based approach for optimizing parallel I/O in FLASH.  
The model predicts MPI I/O performance given simulation parameters and system-level indicators, and suggests the configuration expected to provide the best efficiency.  

---

## How to Run the Demo

1. Execute the script:
   ```bash
   python IO_model.py
    ```
2. Provide the required input values:
    - Number of processors: 4, 8, 16, or 32
    - Blocks per core: 500 or 1000
3. The model will evaluate alternative configurations and return the one predicted to achieve the highest MPI I/O performance. The total number of blocks is kept fixed.

## Example
If the user inputs:
```
Number of cores: 8
Blocks per core: 500
```

The model will compare the performance of:
- 8 cores - 500 blocks
- 4 cores - 1000 blocks
and return the configuration that is predicted to provide superior I/O performance.

## Notes
- The model has been trained on approximately 30,000 runs of the FLASH-IO benchmark.
- Input features include: number of processors, number of aggregators, blocks per core, CPU state indicators (collected via iostat), chassis allocation, time of day and writing mode.
- Results show that independent I/O generally outperforms collective I/O, both in terms of runtime and stability of performance.

## Future Work
- Extend training to full FLASH simulations.
- Incorporate additional system-level performance metrics (e.g., memory bandwidth, network load).
- Investigate online retraining strategies to enable adaptive optimization during runtime.
- Compare the ANN approach with alternative machine learning techniques such as ensemble methods.


