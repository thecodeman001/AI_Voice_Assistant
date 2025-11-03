import argparse
import json
import os
import uuid
from dotenv import load_dotenv

from src.logger import init_logger, LatencyLogger
from src.voice_client import VoiceClient
from src import feedback as feedback_module


def main():
    load_dotenv()
    init_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", choices=["card_lost","transfer_failed","account_locked"], default="card_lost")
    parser.add_argument("--turns", type=int, default=3)
    args = parser.parse_args()

    persona_path = os.path.join("config", "personas", f"{args.persona}.json")
    with open(persona_path, "r") as f:
        persona = json.load(f)

    session_id = str(uuid.uuid4())
    vc = VoiceClient(persona=persona, session_id=session_id)
    logger = LatencyLogger(path=os.path.join("logs", "latency_log.csv"))
    vc.run(max_turns=args.turns, logger_obj=logger, feedback=feedback_module)


if __name__ == "__main__":
    main()
