from phase1.phase1_pipeline import run_phase1
from phase2.phase2_pipeline import run_phase2
from utils.logging_utils import get_logger

logger = get_logger(__name__, log_file="main_pipeline.log")


def main():
    logger.info("========== STARTING COMPLETE BANKING XAI PIPELINE ==========")

    # -----------------------------------------
    # Run Phase-1
    # -----------------------------------------
    logger.info("========== STARTING PHASE-1 ==========")

    phase1_results = run_phase1()

    logger.info("========== PHASE-1 COMPLETED ==========")

    # -----------------------------------------
    # Run Phase-2
    # -----------------------------------------
    logger.info("========== STARTING PHASE-2 ==========")

    run_phase2(
        phase1_results["df"],
        phase1_results["features"],
        phase1_results["target"]
    )

    logger.info("========== PHASE-2 COMPLETED ==========")

    logger.info("========== COMPLETE BANKING XAI PIPELINE COMPLETED SUCCESSFULLY ==========")

    print("\nFull Banking XAI Pipeline Completed Successfully\n")


if __name__ == "__main__":
    main()