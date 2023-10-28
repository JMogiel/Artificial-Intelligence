import random
import time
import matplotlib.pyplot as plt


class Solver:
    def __init__(self, time_quantum=0.1, on_time=10.0, drone_acceleration=30.0, gravity_acceleration=-10.0,
                 drone_friction=-0.1, crash_velocity=20.0, crash_penalty=-1500, population_size=100,
                 mutation_rate=0.1, generations=100):
        # Constants for our problem
        self.time_quantum = time_quantum
        self.on_time = on_time
        self.drone_acceleration = drone_acceleration
        self.gravity_acceleration = gravity_acceleration
        self.drone_friction = drone_friction
        self.crash_velocity = crash_velocity
        self.crash_penalty = crash_penalty
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations

    # listing of parameters
    def get_parameters(self):
        return {
            "time_quantum": self.time_quantum,
            "on_time": self.on_time,
            "drone_acceleration": self.drone_acceleration,
            "gravity_acceleration": self.gravity_acceleration,
            "drone_friction": self.drone_friction,
            "crash_velocity": self.crash_velocity,
            "crash_penalty": self.crash_penalty,
            "population_size": self.population_size,
            "mutation_rate": self.mutation_rate,
            "generations": self.generations
        }

    # simulation of drone flight
    def drone_flight(self, individual):
        # starting values of the drone
        height = 0
        max_height = 0
        drone_velocity = 0
        flight_time = 0
        i = 0
        while height >= 0:  # for i in range(len(individual)):
            if flight_time >= self.on_time:
                drone_velocity += (self.gravity_acceleration - self.drone_friction * abs(
                    drone_velocity)) * self.time_quantum
            elif i < 100:

                if individual[i] == 1:
                    drone_velocity += (self.drone_acceleration + self.gravity_acceleration + self.drone_friction * abs(
                        drone_velocity)) * self.time_quantum
                else:
                    drone_velocity += (self.gravity_acceleration - self.drone_friction * abs(
                        drone_velocity)) * self.time_quantum
                i += 1
            height += drone_velocity * self.time_quantum
            flight_time += 0.1
            max_height = max(height, max_height)
            print(max_height)
        return max_height, abs(drone_velocity), flight_time

    # fitness function for drone_flight
    def fitness_function(self, individual):
        max_height, drone_velocity, flight_time = self.drone_flight(individual)
        if abs(drone_velocity) > self.crash_velocity:
            max_height += self.crash_penalty
        return max_height, abs(drone_velocity), flight_time

    # create individual
    def individual(self):
        return random.choices([0, 1], k=100)

    # create population of individuals
    def population(self):
        return [self.individual() for _ in range(self.population_size)]

    # new individuals from old ones
    def new_individual(self, parent1, parent2):
        where_to_divide = random.randint(1, 99)
        child1 = parent1[:where_to_divide] + parent2[where_to_divide:]
        # child2 = parent2[:where_to_divide] + parent1[where_to_divide:]
        return child1

    # mutate individual by changing "0" and "1"
    def mutate(self, individual):
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] = 1 - individual[i]
        return individual

    # roulette wheel selection
    def roulette_wheel_selection(self, population, fitness_function_value):
        fitness_values = [fitness_function_value(individual)[0] for individual in population]
        numeric_values = [x for x in fitness_values]
        total_fitness = sum(numeric_values)
        selection_probabilities = [fitness / total_fitness for fitness in fitness_values]
        selected_index = random.choices(range(len(population)), weights=selection_probabilities, k=1)[0]
        return population[selected_index]

    # reproduce
    def reproduce(self, population):
        new_population = []
        for i in range(self.population_size):
            parent1 = self.roulette_wheel_selection(population, self.fitness_function)
            parent2 = self.roulette_wheel_selection(population, self.fitness_function)
            child = self.new_individual(parent1, parent2)
            mutated_child = self.mutate(child)
            new_population.append(mutated_child)
        return new_population

    def Solve(self):
        start_time = time.time()
        best_individual = None
        max_fitness_value = float('-inf')
        fitness_history = []
        population_solve = self.population()

        for i in range(self.generations):
            fitness_values = [self.fitness_function(individual)[0] for individual in population_solve]
            for j in range(len(population_solve)):
                if fitness_values[j] > max_fitness_value:
                    max_fitness_value = fitness_values[j]
                    best_individual = population_solve[j]
            population_solve = self.reproduce(population_solve)
            fitness_history.append(max_fitness_value)

        print(f"Best Individual:{best_individual}")
        print(f"Max fitness value:{max_fitness_value}")
        print(f"Velocity for Best Individual:{self.fitness_function(best_individual)[1]}")
        print(f"Time flight for Best Individual:{self.fitness_function(best_individual)[2]}")
        print(f"Time Elapsed:{time.time() - start_time:.2f}s")

        plt.plot(range(self.generations), fitness_history)
        plt.xlabel("Generations")
        plt.ylabel("Best fitness")
        plt.show()


ga = Solver()

sol = ga.Solve()
print("Solve")
print(sol)
