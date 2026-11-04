import os
from pathlib import Path

try:
    import mysql.connector
    from mysql.connector import Error
except ModuleNotFoundError as exc:  # pragma: no cover - dependency error handled at runtime
    mysql = None
    Error = RuntimeError
    MYSQL_IMPORT_ERROR = exc
else:
    MYSQL_IMPORT_ERROR = None

import pandas as pd


TABLE_NAME = "retail_data_c"
COLUMNS = (
    "InvoiceDate",
    "CustomerID",
    "ProductID",
    "ProductName",
    "Category",
    "Quantity",
    "UnitPrice",
    "StockQuantity",
    "Region",
    "Revenue",
)
ENV_FILE = Path(__file__).resolve().with_name(".env")


class DatabaseConnectionError(RuntimeError):
    pass


def load_local_environment() -> None:
    if not ENV_FILE.is_file():
        return

    try:
        lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise DatabaseConnectionError(
            f"Could not read the environment file: {error}"
        ) from error

    for line_number, line in enumerate(lines, start=1):
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        key, separator, value = entry.partition("=")
        if not separator or not key.strip():
            raise DatabaseConnectionError(
                f"Invalid entry in the environment file on line {line_number}."
            )
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def _get_mysql_config() -> dict[str, object]:
    load_local_environment()

    config = {
        "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD"),
        "database": os.getenv("MYSQL_DATABASE", "cusNew"),
        "table_name": os.getenv("MYSQL_TABLE_NAME", TABLE_NAME),
    }

    try:
        import streamlit as st
    except ImportError:
        return config

    try:
        secret_config = st.secrets.get("mysql", {}) if hasattr(st, "secrets") else {}
    except Exception:
        secret_config = {}
    if not config["password"]:
        config["password"] = secret_config.get("password")
    if not config["database"] or config["database"] == "cusNew":
        config["database"] = secret_config.get("database", config["database"])
    if not config["host"] or config["host"] == "127.0.0.1":
        config["host"] = secret_config.get("host", config["host"])
    if not config["user"] or config["user"] == "root":
        config["user"] = secret_config.get("user", config["user"])
    if not config["port"]:
        config["port"] = secret_config.get("port", config["port"])
    if not config["table_name"]:
        config["table_name"] = secret_config.get("table_name", config["table_name"])
    return config


def mysql_is_configured() -> bool:
    config = _get_mysql_config()
    return bool(config["database"] and config["password"])


def load_retail_sales() -> pd.DataFrame:
    config = _get_mysql_config()
    if mysql is None:
        raise DatabaseConnectionError(
            "mysql-connector-python is not installed. Install it with `python -m pip install mysql-connector-python`."
        ) from MYSQL_IMPORT_ERROR
    if not config["password"]:
        raise DatabaseConnectionError(
            f"MYSQL_PASSWORD is not set. Add it to {ENV_FILE} or set it in your environment."
        )
    if not config["database"]:
        raise DatabaseConnectionError(
            "MYSQL_DATABASE is not set. Add it to the environment or Streamlit secrets."
        )

    connection = None
    cursor = None
    try:
        connection = mysql.connector.connect(
            host=config["host"],
            port=int(config["port"]),
            user=config["user"],
            password=config["password"],
            database=config["database"],
        )
        cursor = connection.cursor(dictionary=True)
        columns = ", ".join(f"`{column}`" for column in COLUMNS)
        table_name = config["table_name"]
        cursor.execute(f"SELECT {columns} FROM `{table_name}`")
        rows = cursor.fetchall()
        return pd.DataFrame(rows, columns=COLUMNS)
    except (Error, ValueError) as error:
        raise DatabaseConnectionError(
            f"Could not read the `{config['table_name']}` table from MySQL: {error}"
        ) from error
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


def fetch_walmart_sales() -> pd.DataFrame:
    return load_retail_sales()
