from cobra import Model, Reaction, Metabolite

def build_simple_model():
    model = Model("toy_model")

    # Metabolites
    a = Metabolite("A_c")
    b = Metabolite("B_c")
    c = Metabolite("C_c")

    # Source reaction (external supply of A)
    source = Reaction("SOURCE_A")
    source.lower_bound = 0
    source.upper_bound = 1000
    source.add_metabolites({a: 1})

    # A → B
    r1 = Reaction("R1")
    r1.lower_bound = 0
    r1.upper_bound = 1000
    r1.add_metabolites({a: -1, b: 1})

    # B → C
    r2 = Reaction("R2")
    r2.lower_bound = 0
    r2.upper_bound = 1000
    r2.add_metabolites({b: -1, c: 1})

    # Biomass reaction
    biomass = Reaction("BIOMASS")
    biomass.lower_bound = 0
    biomass.upper_bound = 1000
    biomass.add_metabolites({c: -1})

    model.add_reactions([source, r1, r2, biomass])
    model.objective = "BIOMASS"

    return model