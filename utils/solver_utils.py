import cobra
import logging

logger = logging.getLogger(__name__)

def detect_available_solvers() -> list[str]:
    """Detect available LP solvers supported by cobrapy and optlang."""
    solvers = []
    # Test GLPK
    try:
        import optlang.glpk_interface
        solvers.append("glpk")
    except ImportError:
        pass
    # Test CPLEX
    try:
        import optlang.cplex_interface
        solvers.append("cplex")
    except ImportError:
        pass
    # Test Gurobi
    try:
        import optlang.gurobi_interface
        solvers.append("gurobi")
    except ImportError:
        pass
    # Test HiGHS
    try:
        import optlang.highs_interface
        solvers.append("highs")
    except ImportError:
        pass
    
    # Sane default fallback
    if not solvers:
        solvers = ["glpk"]
    return solvers

def set_solver(model: cobra.Model, solver_name: str) -> None:
    """Set the solver interface on the cobra model."""
    try:
        model.solver = solver_name.lower()
    except Exception as exc:
        logger.error("Failed to set solver %s: %s", solver_name, exc)
        raise exc

def configure_tolerance(model: cobra.Model, feasibility_tol: float, optimality_tol: float) -> None:
    """Configure solver tolerances on the model."""
    try:
        if feasibility_tol is not None:
            model.solver.configuration.tolerances.feasibility = feasibility_tol
        if optimality_tol is not None:
            model.solver.configuration.tolerances.optimality = optimality_tol
    except Exception as exc:
        logger.warning("Failed to configure tolerances: %s", exc)
