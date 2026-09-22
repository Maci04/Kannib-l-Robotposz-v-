# Kannib-l-Robotposz-v-
classDiagram
    class SimulationEngine {
        +int width
        +int height
        +List~Vacuum~ population
        +List~Dirt~ dirts
        +int generation_count
        +run_generation()
        +update_physics()
        -spawn_dirt()
    }

    class Entity {
        <<abstract>>
        +float x
        +float y
        +float radius
        +get_position()
    }

    class Vacuum {
        +float energy
        +float fitness
        +Genome dna
        +act(environment)
        +move(target_x, target_y)
        +attack(target_vacuum)
        +clean(target_dirt)
        -calculate_sensor_data()
    }

    class Dirt {
        +float energy_value
        +bool is_cleaned
    }

    class Genome {
        +float aggression_weight
        +float greed_weight
        +float speed_gene
        +Genome mutate(float rate)
        +Genome crossover(Genome other)
    }

    class EvolutionaryAlgorithm {
        +float mutation_rate
        +int tournament_size
        +List~Vacuum~ evolve(List~Vacuum~ old_population)
        -tournament_selection()
        -calculate_fitness()
    }

    SimulationEngine "1" *-- "*" Vacuum : tartalmaz
    SimulationEngine "1" *-- "*" Dirt : tartalmaz
    Entity <|-- Vacuum : öröklődés
    Entity <|-- Dirt : öröklődés
    Vacuum "1" *-- "1" Genome : kompozíció
    SimulationEngine ..> EvolutionaryAlgorithm : használja
