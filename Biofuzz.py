import torch
import random
import biotorch.applications.adversarial_attacks.functions as attacks
import matplotlib.pyplot as plt

import biotorch.layers.backpropagation as BP
import biotorch.layers.fa as FA
import biotorch.layers.brsf as BRSF
import biotorch.layers.dfa as DFA
import biotorch.layers.frsf as FRSF
import biotorch.layers.usf as USF
import csv

class BioTorchFuzzer:
    def __init__(self, num_iterations, time_objective, bug_discovery_objective):
        self.num_iterations = num_iterations
        self.time_objective = time_objective
        self.bug_discovery_objective = bug_discovery_objective
        self.line_coverage = set()
        self.cumulative_coverage = []
        self.average_coverage = []
        self.target_functions = [
            BP.Linear,
            FA.Linear,
            DFA.Linear,
            BRSF.Linear,
            FRSF.Linear,
            USF.Linear
        ]
        self.mutation_analysis = {
            "total_mutations": 0,
            "successful_mutations": 0,
            "mutation_ratio": 0.0
        }
        self.bug_reports = []

        ### Mutation analysis based on weights
        self.mutation_techniques = [
            {"name": "bit_flipping", "weight": 0.3, "coverage_increase": 0, "bugs_found": 0},
            {"name": "value_replacement", "weight": 0.4, "coverage_increase": 0, "bugs_found": 0},
            {"name": "arithmetic_operations", "weight": 0.3, "coverage_increase": 0, "bugs_found": 0}
        ]

    def generate_input(self):
        # Generate random tensor input
        tensor_size = random.randint(1, 10)
        input_data = torch.randn(tuple([random.randint(1, 5) for _ in range(tensor_size)]))
        return input_data

    def evaluate_mutation_techniques(self):
        # Calculate coverage increase and bugs found for each mutation technique
        for technique in self.mutation_techniques:
            coverage_increase = sum(technique["coverage_increase"] for technique in self.mutation_techniques)
            bugs_found = sum(technique["bugs_found"] for technique in self.mutation_techniques)
            technique["coverage_increase"] = coverage_increase
            technique["bugs_found"] = bugs_found

    def select_mutation_technique(self):
        self.evaluate_mutation_techniques()

        total_weights = sum(technique["weight"] for technique in self.mutation_techniques)
        cumulative_probabilities = [technique["weight"] / total_weights for technique in self.mutation_techniques]

        random_value = random.random()
        selected_technique = None
        cumulative_probability = 0
        for i in range(len(self.mutation_techniques)):
            cumulative_probability += cumulative_probabilities[i]
            if random_value <= cumulative_probability:
                selected_technique = self.mutation_techniques[i]
                break

        return selected_technique

    def analyze_results(self, output):
        for line_num in range(1, torch.numel(output) + 1):
            self.line_coverage.add(line_num)

        selected_technique = self.select_mutation_technique()
        selected_technique["coverage_increase"] += len(self.line_coverage)
        selected_technique["bugs_found"] += len(self.bug_reports)
        
        self.cumulative_coverage.append(len(self.line_coverage))
        self.average_coverage.append(sum(self.cumulative_coverage) / len(self.cumulative_coverage))

    def mutate_input(self, input_data, target_function):
        mutated_input = target_function(input_data)
        self.mutation_analysis["total_mutations"] += 1
        if not torch.all(torch.eq(mutated_input, input_data)):
            self.mutation_analysis["successful_mutations"] += 1

        return mutated_input

    def run(self):
        start_time = time.time()  # Track start time
        for iteration in range(self.num_iterations):
            if time.time() - start_time > self.time_objective:
                break  # Time objective reached, exit loop

            input_data = self.generate_input()

            for target_function in self.target_functions:
                try:
                    model = torch.nn.Sequential(
                        target_function(input_data.size(-1), 16),
                        torch.nn.ReLU(),
                        target_function(16, 8),
                        torch.nn.ReLU(),
                        target_function(8, 1)
                    )
                    output = model(input_data)

                    self.analyze_results(output)

                    input_data = self.mutate_input(input_data, target_function)

                    input_data = self.apply_mutation_operations(input_data)
                    input_data = self.rearrange_data(input_data)

                except Exception as e:
                    self.handle_exception(e, input_data, target_function)

            if len(self.bug_reports) >= self.bug_discovery_objective:
                break  # Bug discovery objective reached, exit loop

        if self.mutation_analysis["total_mutations"] != 0:
            self.mutation_analysis["mutation_ratio"] = (
                self.mutation_analysis["successful_mutations"] / self.mutation_analysis["total_mutations"]
            )
        else:
            self.mutation_analysis["mutation_ratio"] = 0.0


    def handle_exception(self, exception, input_data, target_function):

        error_info = {
            "exception": exception,
            "input_data": input_data,
            "target_function": target_function
        }
        self.bug_reports.append(error_info)

    def apply_mutation_operations(self, input_data):
        selected_technique = self.select_mutation_technique()

        if selected_technique["name"] == "bit_flipping":
            input_data = self.apply_bit_flipping(input_data)
        elif selected_technique["name"] == "value_replacement":
            input_data = self.apply_value_replacement(input_data)
        elif selected_technique["name"] == "arithmetic_operations":
            input_data = self.apply_arithmetic_operations(input_data)

        return input_data


    def apply_bit_flipping(self, input_data, num_bits=5):
        flipped_input = input_data.clone()
        indices = torch.randint(low=0, high=input_data.numel(), size=(num_bits,))
        flipped_input.view(-1)[indices] = 1 - flipped_input.view(-1)[indices]
        return flipped_input

    def apply_value_replacement(self, input_data, num_values=5):
        replaced_input = input_data.clone()
        indices = torch.randint(low=0, high=input_data.numel(), size=(num_values,))
        replaced_input.view(-1)[indices] = torch.randn(num_values)
        return replaced_input

    def apply_arithmetic_operations(self, input_data, num_operations=5):
        mutated_input = input_data.clone()
        for _ in range(num_operations):
            operation = random.choice(["add", "subtract", "multiply", "divide"])
            indices = torch.randint(low=0, high=input_data.numel(), size=(2,))
            operand1 = mutated_input.view(-1)[indices[0]]
            operand2 = mutated_input.view(-1)[indices[1]]
            if operation == "add":
                mutated_input.view(-1)[indices[0]] = operand1 + operand2
            elif operation == "subtract":
                mutated_input.view(-1)[indices[0]] = operand1 - operand2
            elif operation == "multiply":
                mutated_input.view(-1)[indices[0]] = operand1 * operand2
            elif operation == "divide":
                mutated_input.view(-1)[indices[0]] = operand1 / operand2
        return mutated_input

    def rearrange_data(self, input_data):
        input_shape = input_data.shape
        rearranged_data = input_data.flatten().reshape(-1, input_shape[-1])
        random.shuffle(rearranged_data)
        rearranged_data = rearranged_data.reshape(input_shape)
        return rearranged_data

num_iterations = 100
time_objective = 360 
bug_discovery_objective = 500
fuzzer = BioTorchFuzzer(num_iterations, time_objective, bug_discovery_objective)
fuzzer.run()

# Bug Reports
for report in fuzzer.bug_reports:
    print("Exception:", report["exception"])
    #print("Input Data:", report["input_data"])
    print("Target Function:", report["target_function"])
    print()

# Cumulative Coverage
cumulative_coverage = fuzzer.cumulative_coverage
print("Cumulative Coverage:", cumulative_coverage)

# Average Coverage
average_coverage = fuzzer.average_coverage
print("Average Coverage:", average_coverage)

# Mutation Analysis
mutation_ratio = fuzzer.mutation_analysis["mutation_ratio"]
print(f"Mutation Ratio: {mutation_ratio:.2f}")


plt.plot(average_coverage, label = 'average_coverage')
plt.title('Average coverage of cgi_decode() with random inputs')
plt.xlabel('# of inputs')
plt.ylabel('lines covered')

plt.plot(cumulative_coverage, label = 'cumulative_coverage')
plt.title('Coverage of cgi_decode() with random inputs')
plt.xlabel('# of inputs')
plt.ylabel('lines covered')

plt.legend()

new_bug_count = len(fuzzer.bug_reports)
print("Number of new bugs discovered:", new_bug_count)

#####################################################

file_path = "./bug_reports.csv"

with open(file_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    
    writer.writerow(["Exception", "Target Function"])
    
    for report in fuzzer.bug_reports:
        exception = report["exception"]
        target_function = report["target_function"]
        writer.writerow([exception, target_function])
        
print("Bug reports saved to:", file_path)