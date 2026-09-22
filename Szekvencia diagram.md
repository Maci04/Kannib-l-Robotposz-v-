```mermaid

sequenceDiagram
    participant S as SimulationEngine
    participant V as Vacuum (Populáció)
    participant EA as EvolutionaryAlgorithm
    participant G as Genome

    loop Minden szimulációs lépés
        S->>V: act(environment)
        V-->>S: state_updated
    end
    
    Note over S,V: Generáció vége (mindenki lemerült)
    
    S->>EA: evolve(current_population)
    activate EA
    
    EA->>EA: calculate_fitness()
    EA->>EA: tournament_selection()
    
    loop Új populáció méretéig
        EA->>G: crossover(parent1, parent2)
        G-->>EA: child_genome
        EA->>G: mutate(child_genome, rate)
        G-->>EA: mutated_genome
    end
    
    EA-->>S: new_population
    deactivate EA
    
    S->>S: reset_environment()
    S->>S: run_generation(new_population)
