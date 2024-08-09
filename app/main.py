import logging
from pipeline_engine import PipelineEngine
import time
import signal
import sys
import asyncio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def signal_handler(sig, frame):
    logger.info("Received signal to terminate. Exiting gracefully...")
    sys.exit(0)

async def main(config_path: str):
    pipeline = PipelineEngine(config_path)
    await pipeline.run()
    
    logger.info("Pipeline execution completed. Entering stall mode...")
    
    # Register the signal handler for graceful termination
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        while True:
            # This keeps the program running indefinitely
            await asyncio.sleep(60)  # Sleep for 60 seconds before checking again
            logger.debug("Application is still running in stall mode...")
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Exiting gracefully...")
    finally:
        logger.info("Application shutting down.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_config.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    asyncio.run(main(config_path))