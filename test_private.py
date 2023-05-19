import unittest
import numpy as np
from dynprog import DroneExtinguisher

class MyTestCase(unittest.TestCase):
    def test_dynamic_programming_multiple_drones_mixed_order(self):
        forest_location = (0, 0)
        bags = [3, 9, 2, 3, 19]
        bag_locations = [(3, 4) for _ in range(len(bags))]  # constant travel distance 5
        liter_cost_per_km = 0.1
        liter_budget_per_day = 20
        usage_cost = np.array([[1, 1, 0],
                               [1, 1, 0],
                               [1, 1, 0],
                               [1, 1, 0],
                               [0, 1, 1]]) #should still pick drone 2

        solution = 2414

        de = DroneExtinguisher(
            forest_location=forest_location,
            bags=bags,
            bag_locations=bag_locations,
            liter_cost_per_km=liter_cost_per_km,
            liter_budget_per_day=liter_budget_per_day,
            usage_cost=usage_cost
        )

        de.fill_travel_costs_in_liters()
        de.dynamic_programming()
        lowest_cost = de.lowest_cost()
        self.assertEqual(lowest_cost, solution)

    def test_backtrace_memory_simple(self):
        forest_location = (0, 0)
        bags = [3, 9, 2, 3, 19]
        bag_locations = [(3, 4) for _ in range(len(bags))]  # constant travel distance 5
        liter_cost_per_km = 0.1
        liter_budget_per_day = 20
        usage_cost = np.array([[0, 1, 1],
                               [0, 1, 1],
                               [1, 0, 1],
                               [1, 0, 1],
                               [1, 1, 0]])

        solution = ([0, 2, 4], [0, 0, 1, 1, 2])

        de = DroneExtinguisher(
            forest_location=forest_location,
            bags=bags,
            bag_locations=bag_locations,
            liter_cost_per_km=liter_cost_per_km,
            liter_budget_per_day=liter_budget_per_day,
            usage_cost=usage_cost
        )

        de.fill_travel_costs_in_liters()
        de.dynamic_programming()
        self.assertEqual(solution, de.backtrace_solution())

    def test_dynamic_programming_varying_distance(self):
        forest_location = (0, 0)
        bags = [3, 4, 6, 5, 3, 4]
        bag_locations = [(3, 4), (5, 0), (0, 1), (1, 0), (5, 0), (2, 0)]
        liter_cost_per_km = 1
        liter_budget_per_day = 35
        usage_cost = np.array([[0, 2, 0],
                               [0, 0, 1],
                               [0, 6, 0],
                               [4, 2, 1],
                               [3, 0, 4],
                               [3, 4, 1]])
        solution = 6
        de = DroneExtinguisher(forest_location=forest_location, bags=bags, bag_locations=bag_locations,
                               liter_cost_per_km=liter_cost_per_km,        liter_budget_per_day=liter_budget_per_day,
                               usage_cost=usage_cost)
        de.fill_travel_costs_in_liters()
        de.dynamic_programming()
        lowest_cost = de.lowest_cost()
        print(de.optimal_cost)
        print(de.backtrace_memory)
        print(de.backtrace_solution())
        self.assertEqual(lowest_cost, solution)

if __name__ == '__main__':
    unittest.main()
