from config import DATABASE_PATH
from modules.database import SessionDatabase


def main():
    database = SessionDatabase(DATABASE_PATH)
    try:
        sessions = database.recent_sessions(limit=20)
        if not sessions:
            print("No sessions found yet.")
            return

        print("Recent sessions:")
        for session_id, start_time, end_time, final_sentence in sessions:
            print(f"{session_id}: {start_time} -> {end_time or 'active'} | {final_sentence or ''}")
    finally:
        database.close()


if __name__ == "__main__":
    main()
