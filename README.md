## BioTorch 
```pip install biotorch```
Or you can get sourc
```git clone https://github.com/jsalbert/biotorch.git```
```cd biotorch```
```script/setup```
# BioTorch Fuzzer

BioTorch Fuzzer is a tool for testing and analyzing code coverage and bug detection of various functions provided by the BioTorch library. It allows for effective testing and debugging of BioTorch functions.

## Usage

1. Install Required Libraries

   To run BioTorch Fuzzer, you need the following libraries:
   - torch: The PyTorch library
   - random: Library for random number generation
   - matplotlib.pyplot: Library for graph visualization

   Install the required libraries.

2. Run BioTorch Fuzzer

   Create an instance of the BioTorchFuzzer class and call the `run` method to execute the test. You can specify the number of iterations to run the test by setting the `num_iterations` parameter.

   ```python
   num_iterations = 100
   fuzzer = BioTorchFuzzer(num_iterations)
   fuzzer.run() ```

   After running the test, you can see the following results:
   
   - Bug Reports: Displays information about discovered bugs
   - Cumulative Coverage: Prints the cumulative value of code coverage
   - Average Coverage: Prints the average value of code coverage
   - Mutation Ratio: Prints the mutation ratio

   3. Graph Visualization
   You can use the matplotlib.pyplot library to visualize the coverage graph. Use the average_coverage and cumulative_coverage lists to plot the graph.

   ```import matplotlib.pyplot as plt

plt.plot(average_coverage)
plt.title('Average coverage of BioTorch functions with random inputs')
plt.xlabel('# of inputs')
plt.ylabel('lines covered')

plt.plot(cumulative_coverage)
plt.title('Coverage of BioTorch functions with random inputs')
plt.xlabel('# of inputs')
plt.ylabel('lines covered')
```


