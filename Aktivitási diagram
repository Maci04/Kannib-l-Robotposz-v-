```mermaid

stateDiagram-v2
    [*] --> SzenzorAdatokOlvasása : Ciklus kezdete
    SzenzorAdatokOlvasása --> Döntéshozatal : Környezet felmérése
    
    state Döntéshozatal {
        [*] --> EnergiaszintEllenőrzés
        EnergiaszintEllenőrzés --> CélpontKeresés : Energia > Kritikus
        EnergiaszintEllenőrzés --> TúlélőMód : Energia <= Kritikus
        
        CélpontKeresés --> KoszCélbavétele : Kosz van közelebb / Nagy greed_weight
        CélpontKeresés --> PorszívóCélbavétele : Másik ágens van közelebb / Nagy aggression_weight
    }
    
    Döntéshozatal --> Mozgás
    Mozgás --> AkcióVégrehajtása : Célpont elérve
    Mozgás --> EnergiaCsökkentése : Lépés megtörtént
    
    state AkcióVégrehajtása {
        KoszFeltakarítása --> FitneszNövelése
        PorszívóTámadása --> EnergiaLecsapolása
    }
    
    AkcióVégrehajtása --> Halál : Energia elfogyott
    AkcióVégrehajtása --> [*] : Ciklus vége
    EnergiaCsökkentése --> Halál : Energia = 0
    EnergiaCsökkentése --> [*]
    Halál --> [*]
