import sqlite3
from datetime import datetime

class ExperimentLogger:
    def __init__(self, db_path="experiments.db"):
        self.conn = sqlite3.connect(db_path)
        self.create_tables()

    def create_tables(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS LLAMAExperiments (
            id INTEGER PRIMARY KEY,
            model_name TEXT,
            lora_config TEXT,
            train_loss REAL,
            val_loss REAL,
            metrics TEXT,
            timestamp TEXT
        )
        """)

    def log_experiment(self, model_name, lora_config, train_loss, val_loss, metrics):
        self.conn.execute("""
        INSERT INTO LLAMAExperiments VALUES (NULL,?,?,?,?,?,?)
        """, (
            model_name,
            str(lora_config),
            train_loss,
            val_loss,
            str(metrics),
            datetime.now().isoformat()
        ))
        self.conn.commit()
